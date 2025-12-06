# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdMolAlign

from .constants import PAD_ID, MAX_ATOMS


def _safe_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """从 SMILES 构建并 sanitize 分子，失败返回 None。"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def _embed_3d(mol: Chem.Mol) -> bool:
    """
    使用 ETKDGv3 + UFF 优化生成 3D 构象。
    成功返回 True，失败返回 False。
    """
    try:
        params = rdDistGeom.ETKDGv3()
        params.useRandomCoords = True
        cid = AllChem.EmbedMolecule(mol, params)
        if cid != 0:
            return False
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            # UFF 优化失败不致命，只要有 3D 构象即可
            pass
        return True
    except Exception:
        return False


def mol_from_smiles_3d(smiles: str, add_hs: bool = True) -> Optional[Chem.Mol]:
    """
    从 SMILES 得到带 3D conformer 的 RDKit Mol，用于 RMSD 计算 / 结构对齐。
    失败返回 None。
    """
    mol = _safe_mol_from_smiles(smiles)
    if mol is None:
        return None

    if add_hs:
        try:
            mol = Chem.AddHs(mol)
        except Exception:
            # 加氢失败也不必直接报错，尽量继续
            pass

    ok = _embed_3d(mol)
    if (not ok) or (mol.GetNumConformers() == 0):
        return None
    return mol


def smiles_to_graph(
    smiles: str,
    max_atoms: int = MAX_ATOMS,
    add_hs: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    把 SMILES 转为 (atom_ids, coords, pad_mask)

      - atom_ids: (N,) int64，PAD 位置为 PAD_ID(=0)
      - coords  : (N,3) float32，PAD 位置为 0
      - pad_mask: (N,) bool，True 表示 PAD

    N = max_atoms；若分子超过则截断，少于则补齐。
    """
    mol = _safe_mol_from_smiles(smiles)
    if mol is None:
        # 完全无效：返回全 PAD
        atom_ids = torch.full((max_atoms,), PAD_ID, dtype=torch.long)
        coords = torch.zeros(max_atoms, 3, dtype=torch.float32)
        pad = torch.ones(max_atoms, dtype=torch.bool)
        return atom_ids, coords, pad

    # 可选加氢（更稳定的 3D 构象）
    if add_hs:
        try:
            mol = Chem.AddHs(mol)
        except Exception:
            pass

    has_conf = _embed_3d(mol)
    n_atoms = mol.GetNumAtoms()
    n_keep = min(n_atoms, max_atoms)

    # 原子序号（作为“词 id”）
    Z = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n_atoms)]
    Z = np.asarray(Z[:n_keep], dtype=np.int64)

    # 坐标
    if has_conf and mol.GetConformer().Is3D():
        conf = mol.GetConformer()
        xyz = np.zeros((n_keep, 3), dtype=np.float32)
        for i in range(n_keep):
            p = conf.GetAtomPosition(i)
            xyz[i, 0], xyz[i, 1], xyz[i, 2] = float(p.x), float(p.y), float(p.z)
    else:
        # 没有 3D 时用 0（注意这类样本在 RMSD 命中时自然会被判为不命中）
        xyz = np.zeros((n_keep, 3), dtype=np.float32)

    # 组装到固定长度
    atom_ids = np.full((max_atoms,), PAD_ID, dtype=np.int64)
    coords = np.zeros((max_atoms, 3), dtype=np.float32)
    pad = np.ones((max_atoms,), dtype=bool)

    atom_ids[:n_keep] = Z
    coords[:n_keep] = xyz
    pad[:n_keep] = False

    return (
        torch.from_numpy(atom_ids),
        torch.from_numpy(coords),
        torch.from_numpy(pad),
    )


# ===== 3D 结构对齐与 RMSD 命中判定 =====

def rmsd_between_smiles(
    smiles_pred: str,
    smiles_ref: str,
    add_hs: bool = True
) -> Optional[float]:
    """
    对预测分子和真实分子进行 3D 刚体对齐，并返回最优 RMSD（Å）。

    步骤：
      1. 从 SMILES 构建两个 3D 分子 (mol_pred, mol_ref)；
      2. 若原子数不同，直接返回 None（视作不匹配）；
      3. 用 rdMolAlign.GetBestRMS(mol_ref, mol_pred) 求最优对齐 RMSD。

    返回：
      - 成功: float (单位 Å)
      - 失败: None
    """
    mol_pred = mol_from_smiles_3d(smiles_pred, add_hs=add_hs)
    mol_ref  = mol_from_smiles_3d(smiles_ref,  add_hs=add_hs)

    if mol_pred is None or mol_ref is None:
        return None

    if mol_pred.GetNumAtoms() != mol_ref.GetNumAtoms():
        # 原子数不同，说明拓扑差别较大，不再强行对齐
        return None

    try:
        # GetBestRMS 会在所有对齐方式（考虑对称性）中找到 RMSD 最小的那个
        rms = float(rdMolAlign.GetBestRMS(mol_ref, mol_pred))
        return rms
    except Exception:
        return None


def is_rmsd_hit(
    smiles_pred: str,
    smiles_ref: str,
    thresh: float = 1.25,
    add_hs: bool = True
) -> bool:
    """
    基于 3D RMSD 的“命中”判定：
      - 若 RMSD <= thresh (Å)，视为预测分子命中真实分子；
      - 否则 / 或 RMSD 计算失败，则视为未命中。
    """
    rms = rmsd_between_smiles(smiles_pred, smiles_ref, add_hs=add_hs)
    return (rms is not None) and (rms <= thresh)
