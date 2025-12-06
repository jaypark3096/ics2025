# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from .featurizer import mol_from_smiles_3d
from .constants import RMSD_DEFAULT_THRESH


def _build_mol_cache(
    smiles_lists: List[List[str]]
) -> Dict[str, Optional[Chem.Mol]]:
    """
    从多个 SMILES 列表构建缓存：smi -> 带 3D 构象的 RDKit Mol / None
    （避免重复做 SMILES 解析 + 3D 构象生成）。
    """
    cache: Dict[str, Optional[Chem.Mol]] = {}
    for lst in smiles_lists:
        for smi in lst:
            if not isinstance(smi, str):
                continue
            s = smi.strip()
            if not s or s in cache:
                continue
            cache[s] = mol_from_smiles_3d(s)  # 失败会返回 None
    return cache


def topk_and_coverage(
    pred_lists: List[List[str]],
    gt_lists: List[List[str]],
    k: int = 5,
    rmsd_threshold: float = RMSD_DEFAULT_THRESH,
) -> Dict[str, float]:
    """
    基于 3D 结构 RMSD 的 MetaTrans 风格 top-k 评估。

    参数:
      pred_lists:  List[预测列表]，每个元素是一个样本的预测 metabolite SMILES 列表，
                   已按从好到坏排序；只用前 k 个。
      gt_lists:    List[真实列表]，每个元素是该样本的真实 metabolite SMILES 列表。
      k:           使用前 k 个预测。
      rmsd_threshold:
                   3D RMSD 阈值，<= 该阈值则认为“命中”（单位 Å）。
                   默认使用 constants.RMSD_DEFAULT_THRESH（通常 1.0 或 1.25）。

    命中定义（单个样本，基于 3D 结构对齐）：
      - 对每个真实代谢物 g：
          * 在所有“尚未使用”的预测 p 中，找到 RMSD 最小的那个；
          * 若 min_rmsd <= rmsd_threshold，则记为命中一次，
            且该预测 p 不能再匹配其他真实（贪心一对一匹配）。
      - RMSD 通过 rdMolAlign.GetBestRMS(g, p) 获得，内部自动做刚体对齐。
      - 若任一分子 3D 构象生成失败或原子数不同，则这对 (g,p) 不参与匹配。
    """
    assert len(pred_lists) == len(gt_lists)

    # 预建 3D 分子缓存
    mol_cache = _build_mol_cache(pred_lists + gt_lists)

    n_valid = 0

    atleast_one = 0
    atleast_half = 0
    all_cover = 0

    per_precisions: List[float] = []
    per_recalls: List[float] = []

    total_correct = 0  # 所有样本命中的真实代谢物总数
    total_gt = 0       # 所有样本真实代谢物总数

    for preds_all, gt_all in zip(pred_lists, gt_lists):
        # ---- 真实 SMILES 去重并过滤掉无法生成 3D 的 ----
        gt_unique = list(dict.fromkeys(gt_all))  # 去重但保留顺序
        gt_mols = []
        for smi in gt_unique:
            mol = mol_cache.get(smi, None)
            if mol is not None and mol.GetNumAtoms() > 0:
                gt_mols.append((smi, mol))

        if len(gt_mols) == 0:
            # 该样本所有真实 SMILES 都解析/3D 生成失败，跳过
            continue

        n_gt = len(gt_mols)
        total_gt += n_gt
        n_valid += 1

        # ---- 预测 SMILES：只取 top-k，并过滤掉 3D 失败的 ----
        preds_topk = preds_all[:k]
        pred_mols: List[Optional[Chem.Mol]] = []
        for smi in preds_topk:
            mol = mol_cache.get(smi, None)
            if mol is not None and mol.GetNumAtoms() > 0:
                pred_mols.append(mol)
            else:
                pred_mols.append(None)  # 占位，后面会跳过

        used_pred_idx = set()
        n_hit = 0

        # ---- 对每个真实代谢物做一对一贪心匹配（以 RMSD 最小为准） ----
        for g_smi, g_mol in gt_mols:
            if g_mol is None or g_mol.GetNumAtoms() == 0:
                continue

            best_j = None
            best_rmsd = float("inf")

            for j, p_mol in enumerate(pred_mols):
                if j in used_pred_idx:
                    continue
                if p_mol is None:
                    continue
                # 原子数不同则直接跳过
                if p_mol.GetNumAtoms() != g_mol.GetNumAtoms():
                    continue

                try:
                    # GetBestRMS 会先做刚体对齐再计算 RMSD
                    rmsd = float(rdMolAlign.GetBestRMS(g_mol, p_mol))
                except Exception:
                    continue

                if rmsd < best_rmsd:
                    best_rmsd = rmsd
                    best_j = j

            if best_j is not None and best_rmsd <= rmsd_threshold:
                n_hit += 1
                used_pred_idx.add(best_j)

        # 样本级统计
        total_correct += n_hit

        # At least one / half / all
        if n_hit >= 1:
            atleast_one += 1
        if n_hit >= max(1, int(np.ceil(n_gt / 2.0))):
            atleast_half += 1
        if n_hit == n_gt:
            all_cover += 1

        # Precision / Recall（样本级）
        num_pred_used = max(1, len(preds_topk))
        prec = n_hit / num_pred_used
        rec = n_hit / n_gt
        per_precisions.append(prec)
        per_recalls.append(rec)

    if n_valid == 0 or total_gt == 0:
        return {
            "AtLeastOne": 0.0,
            "AtLeastHalf": 0.0,
            "All": 0.0,
            "TotalIdentified": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
        }

    return {
        "AtLeastOne": 100.0 * atleast_one / n_valid,
        "AtLeastHalf": 100.0 * atleast_half / n_valid,
        "All": 100.0 * all_cover / n_valid,
        "TotalIdentified": 100.0 * total_correct / total_gt,
        "Precision": 100.0 * float(np.mean(per_precisions)),
        "Recall": 100.0 * float(np.mean(per_recalls)),
    }
