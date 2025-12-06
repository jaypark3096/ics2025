# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .paths import CSV_PATH, ENZ_DIR
from .constants import PAD_ID, MAX_ATOMS
from .featurizer import smiles_to_graph


def _detect_enzyme_dim() -> int:
    """
    在 ENZ_DIR 中找一个 .npy 文件，推断酶特征维度 D。
    支持两种格式：
      - (L, D): 序列特征
      - (D,)  : 全局特征
    """
    assert ENZ_DIR.is_dir(), f"酶特征目录不存在: {ENZ_DIR}"
    for fn in os.listdir(ENZ_DIR):
        if fn.endswith(".npy"):
            arr = np.load(ENZ_DIR / fn, mmap_mode="r")
            if arr.ndim == 2:   # (L, D)
                return int(arr.shape[1])
            elif arr.ndim == 1: # (D,)
                return int(arr.shape[0])
    raise RuntimeError(f"在 {ENZ_DIR} 中未发现可用的 .npy 酶特征文件")


def _load_enzyme_matrix(enzyme_id: str | None, enz_dim: int) -> torch.Tensor:
    """
    返回形状 (L, D) 的酶特征矩阵：
      - 若磁盘文件是 (L, D) 直接读取
      - 若是 (D,) 则扩成 (1, D)
      - 若缺失/None，返回 (1, D) 的全零
    """
    if not enzyme_id:
        return torch.zeros(1, enz_dim, dtype=torch.float32)

    p = ENZ_DIR / f"{enzyme_id}.npy"
    if not p.is_file():
        return torch.zeros(1, enz_dim, dtype=torch.float32)

    arr = np.load(p)
    if arr.ndim == 1:      # (D,)
        arr = arr[None, :] # -> (1, D)
    elif arr.ndim != 2:
        raise ValueError(f"不支持的酶特征形状 {arr.shape} @ {p}")
    return torch.from_numpy(arr.astype(np.float32))  # (L, D)


class MetabolicDataset(Dataset):
    """
    每条样本 = (Reaction_ID, 底物(所有底物融合) SMILES, 单个产物 SMILES, 酶 ID)

    一个 Reaction_ID 有多个产物时，会展开成多条样本：
      (rid, sub_merged, prod1), (rid, sub_merged, prod2), ...

    底物部分：同一 Reaction 所有底物 SMILES 去重后用 '.' 连接，再转成 3D 图，
    以此同时利用所有底物且保持维度一致。

    train / eval 划分在 Reaction_ID 粒度上进行：
      - 先对 Reaction_ID 做划分
      - 再按 rid 把对应的 (rid, prod*) 样本分配到 train 或 eval

    如果传入 allowed_rids，则完全由外部控制使用哪些 Reaction_ID，
    不再使用 eval_ratio 做随机划分，这样方便做 5-fold 交叉验证。
    """

    def __init__(
        self,
        csv_path: Path = CSV_PATH,
        split: str = "train",
        eval_ratio: float = 0.1,
        random_state: int = 42,
        allowed_rids: Optional[Sequence[str]] = None,
    ):
        csv_path = Path(csv_path)
        print(f"[Data] 读取: {csv_path}")
        assert csv_path.is_file(), f"CSV不存在: {csv_path}"

        # 尝试常见编码，避免 Windows 下 UTF-8 报错
        encodings = ["utf-8", "utf-8-sig", "latin1", "gbk", "cp1252"]
        last_err = None
        for enc in encodings:
            try:
                self.df = pd.read_csv(csv_path, encoding=enc)
                print(f"[Data] 使用编码 {enc} 读取 CSV 成功")
                last_err = None
                break
            except Exception as e:
                last_err = e
        if last_err is not None:
            raise last_err

        self.enzyme_dim = _detect_enzyme_dim()

        # ---------- 先按 Reaction_ID 聚合，再展开到每个产物 ----------
        items: List[Dict] = []
        for rid, g in self.df.groupby("Reaction_ID"):
            roles = g["Metabolite_Role"].astype(str).str.lower()

            # 底物行
            subs = g[roles.isin(["substrate", "reactant", "sub"])]
            # 产物行
            prods = g[roles.isin(["product", "metabolite", "prod", "product_molecule"])]

            if len(subs) == 0 or len(prods) == 0:
                continue

            # === 底物：利用所有底物，去重后用 '.' 融合成一个多组分 SMILES ===
            sub_smiles_list = (
                subs["Metabolite_SMILES"]
                .dropna()
                .astype(str)
                .tolist()
            )
            sub_smiles_list = sorted(set(sub_smiles_list))
            if len(sub_smiles_list) == 0:
                continue
            # 多底物融合为一个 SMILES：smi1.smi2.smi3
            sub_smiles_merged = ".".join(sub_smiles_list)

            # 该反应下所有不同产物 SMILES（去重）
            prod_smiles_all = (
                prods["Metabolite_SMILES"]
                .dropna()
                .astype(str)
                .tolist()
            )
            prod_smiles_all = sorted(set(prod_smiles_all))
            if len(prod_smiles_all) == 0:
                continue

            # === 酶 ID：每个 Reaction_ID 对应一个 Enzyme_HMDB_ID ===
            enz_ids = (
                g["Enzyme_HMDB_ID"]
                .dropna()
                .astype(str)
                .unique()
            )
            enzyme_id: Optional[str]
            if len(enz_ids) == 0:
                enzyme_id = None
            elif len(enz_ids) == 1:
                enzyme_id = str(enz_ids[0])
            else:
                # 一个反应出现了多个 Enzyme_HMDB_ID，打印 warning 方便检查
                print(f"[Warn] Reaction_ID={rid} 对应多个 Enzyme_HMDB_ID={list(enz_ids)}，仅使用第一个 {enz_ids[0]}")
                enzyme_id = str(enz_ids[0])

            # 展开成多条样本 (rid, sub_merged, prod_j)
            # 同时在每条样本里附带该反应的所有产物列表，方便后续 top-k 评估使用
            for prod_smiles in prod_smiles_all:
                items.append(dict(
                    reaction_id=str(rid),
                    sub_smiles=sub_smiles_merged,
                    prod_smiles=prod_smiles,
                    enzyme_id=enzyme_id,
                    all_prod_smiles=prod_smiles_all,
                ))

        # ---------- 按 Reaction_ID 做 train/eval 划分 ----------

        rng = np.random.default_rng(random_state)
        all_rids = sorted({it["reaction_id"] for it in items})

        # 如果外部明确指定了要用哪些 Reaction_ID（例如 5-fold 交叉验证）
        if allowed_rids is not None:
            allowed_rids_set = {str(r) for r in allowed_rids}
            all_rids = [rid for rid in all_rids if rid in allowed_rids_set]

            pick = [it for it in items if it["reaction_id"] in allowed_rids_set]
        else:
            rng.shuffle(all_rids)

            if eval_ratio < 0.0 or eval_ratio > 1.0:
                raise ValueError(f"eval_ratio 必须在 [0,1]，当前为 {eval_ratio}")

            n_eval_r = int(len(all_rids) * eval_ratio)
            eval_rids = set(all_rids[:n_eval_r])
            train_rids = set(all_rids[n_eval_r:])

            if split == "train":
                pick = [it for it in items if it["reaction_id"] in train_rids]
            elif split in ("eval", "val", "valid", "test"):
                pick = [it for it in items if it["reaction_id"] in eval_rids]
            else:
                raise ValueError(f"未知 split={split}, 仅支持 train / eval")

        self.items = pick
        print(f"[Data] split={split}, Reaction_ID 数={len(all_rids)}, 样本数={len(self.items)}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor | str]:
        it = self.items[i]

        # 底物（所有底物融合后的 SMILES）
        atom_sub, xyz_sub, pad_sub = smiles_to_graph(it["sub_smiles"], max_atoms=MAX_ATOMS)
        # 单个产物
        atom_prod, xyz_prod, pad_prod = smiles_to_graph(it["prod_smiles"], max_atoms=MAX_ATOMS)
        # 酶
        enzyme = _load_enzyme_matrix(it["enzyme_id"], self.enzyme_dim)

        return dict(
            reaction_id=it["reaction_id"],
            atom_sub=atom_sub, xyz_sub=xyz_sub, pad_sub=pad_sub,
            atom_prod=atom_prod, xyz_prod=xyz_prod, pad_prod=pad_prod,
            enzyme=enzyme,
            # all_prod_smiles 保留在 Python 端用即可，collate_fn 里目前先不合并这个字段
            all_prod_smiles=it.get("all_prod_smiles", None),
        )


def collate_fn(batch: List[Dict], pad_id: int = PAD_ID, max_atoms: int = MAX_ATOMS) -> Dict[str, torch.Tensor]:
    """
    分子: stack；酶: 按当前 batch 的 L_max pad 到同一长度。
    """
    # 底物
    atom_sub = torch.stack([x["atom_sub"] for x in batch], 0)  # (B,N)
    xyz_sub  = torch.stack([x["xyz_sub"]  for x in batch], 0)  # (B,N,3)
    pad_sub  = torch.stack([x["pad_sub"]  for x in batch], 0)  # (B,N)

    # 产物
    atom_prod = torch.stack([x["atom_prod"] for x in batch], 0)
    xyz_prod  = torch.stack([x["xyz_prod"]  for x in batch], 0)
    pad_prod  = torch.stack([x["pad_prod"]  for x in batch], 0)

    # 酶：变长 → 当前 batch 的 L_max
    Ls = [x["enzyme"].shape[0] for x in batch]
    L_max = max(Ls)
    D = batch[0]["enzyme"].shape[1]
    B = len(batch)

    enzyme = torch.zeros(B, L_max, D, dtype=torch.float32)
    enzyme_mask = torch.zeros(B, L_max, dtype=torch.bool)
    for b, x in enumerate(batch):
        L = x["enzyme"].shape[0]
        enzyme[b, :L] = x["enzyme"]
        enzyme_mask[b, :L] = True

    # 对齐 train.py / run_finetune.py 里期望的键名
    return dict(
        atom_s=atom_sub, xyz_s=xyz_sub, pad_s=pad_sub,
        atom_p=atom_prod, xyz_p=xyz_prod, pad_p=pad_prod,
        enzyme=enzyme, enzyme_mask=enzyme_mask,
    )
