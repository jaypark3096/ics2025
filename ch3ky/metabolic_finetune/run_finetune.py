# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from tqdm.auto import tqdm
from rdkit import RDLogger
import random
import os
import time  # 计时用

# 让 “python run_finetune.py” 和 “python -m metabolic_finetune.run_finetune” 都能工作
PKG_DIR = Path(__file__).resolve().parent
ROOT_DIR = PKG_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from metabolic_finetune.paths import (  # type: ignore
    CSV_PATH, OUT_DIR, ENC_PATH, DEC_PATH, PRED_DIR
)
from metabolic_finetune.constants import (  # type: ignore
    PAD_ID, BATCH_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY,
    NOISE_STD, MAX_ATOMS, NUM_WORKERS, DEVICE, SEED,
    D_LR, D_STEPS, LAMBDA_ADV, LAMBDA_TRI, RMSD_DEFAULT_THRESH,
)
from metabolic_finetune.dataset import MetabolicDataset, collate_fn  # type: ignore
from metabolic_finetune.metabolic_model import MetabolicFinetuneModel  # type: ignore
from metabolic_finetune.discriminators import MultiTaskDiscriminator  # type: ignore

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

# 早停相关超参数
EARLY_STOPPING_PATIENCE = 10   # 连续多少个 epoch 无提升则提前停止
EARLY_STOPPING_DELTA = 0.0     # 要求 val_loss 至少降低多少才算“有提升”

# 单个 batch 的最大允许耗时（秒）——10 分钟
MAX_BATCH_SECONDS = 600.0

# GAN warmup：前多少个 epoch 只用监督重建，不启用判别器 / 对抗损失
GAN_WARMUP_EPOCHS = 2


def _format_rids_for_log(rids) -> str:
    """把 batch 里的 reaction_id 打印得稍微好一点，太长就截断。"""
    try:
        r_list = list(rids)
    except TypeError:
        r_list = [rids]
    r_list = [str(x) for x in r_list]
    if len(r_list) > 10:
        return f"{r_list[:10]} ... (total {len(r_list)})"
    else:
        return str(r_list)


def _read_csv_with_encoding(csv_path: Path) -> pd.DataFrame:
    """
    兼容 merged_HMDB_drugbank_reactions.csv 的各种编码问题。
    依次尝试 utf-8, utf-8-sig, latin1, gbk, cp1252。
    """
    encodings = ["utf-8", "utf-8-sig", "latin1", "gbk", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            print(f"[CSV] 使用编码 {enc} 读取 {csv_path.name} 成功")
            last_err = None
            return df
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"无法读取 CSV: {csv_path}")


def _build_reaction_gt(csv_path: Path) -> Dict[str, List[str]]:
    """
    从原始 CSV 构建：
      - reaction_to_gt: Reaction_ID -> 真实产物 SMILES 列表（去重）

    注意：这里只是为了在 CSV 中展示真实代谢物的 SMILES；
    真正的 top-k 命中是基于 3D 结构 + RMSD，而不是指纹。
    """
    df = _read_csv_with_encoding(csv_path)

    roles = df["Metabolite_Role"].astype(str).str.lower()
    prod_flags = roles.isin(
        ["product", "metabolite", "prod", "product_molecule"]
    )

    reaction_to_gt: Dict[str, List[str]] = {}

    for rid, g in df.groupby("Reaction_ID"):
        g_prod = g[prod_flags.loc[g.index]]
        if g_prod.empty:
            continue
        smi_list = g_prod["Metabolite_SMILES"].astype(str).tolist()
        reaction_to_gt[str(rid)] = sorted(set(smi_list))

    return reaction_to_gt


def _load_ckpt_or_die(model: MetabolicFinetuneModel):
    """
    从化学反应预训练阶段加载共享编码器/解码器权重。
    假定 ENC_PATH / DEC_PATH 已由 paths.py 解析好。
    """
    assert ENC_PATH.is_file(), f"共享编码器缺失: {ENC_PATH}"
    assert DEC_PATH.is_file(), f"共享解码器缺失: {DEC_PATH}"

    enc_state = torch.load(str(ENC_PATH), map_location="cpu", weights_only=False)
    if isinstance(enc_state, dict) and "state_dict" in enc_state:
        enc_state = enc_state["state_dict"]

    dec_state = torch.load(str(DEC_PATH), map_location="cpu", weights_only=False)
    if isinstance(dec_state, dict) and "state_dict" in dec_state:
        dec_state = dec_state["state_dict"]

    model.load_shared_encoder(enc_state)
    model.load_shared_decoder(dec_state)
    print(f"[Info] 加载共享编码器: {ENC_PATH}")
    print(f"[Info] 加载共享解码器: {DEC_PATH}")


def _save_checkpoint(
    fold_id: int,
    epoch: int,
    model: MetabolicFinetuneModel,
    disc: MultiTaskDiscriminator,
    opt_G: AdamW,
    opt_D: AdamW,
):
    """保存当前 fold 的 checkpoint（每个 epoch 结束调用一次）"""
    ckpt_path = OUT_DIR / f"checkpoint_fold{fold_id + 1}.pt"

    ckpt = {
        "fold_id": fold_id,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "disc_state_dict": disc.state_dict(),
        "opt_G_state_dict": opt_G.state_dict(),
        "opt_D_state_dict": opt_D.state_dict(),
        "rng_state": {  # 不再恢复，只做记录
            "random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(ckpt, ckpt_path)
    print(f"[Fold {fold_id + 1}] 已保存 checkpoint: {ckpt_path}")


def _try_load_checkpoint(
    fold_id: int,
    model: MetabolicFinetuneModel,
    disc: MultiTaskDiscriminator,
    opt_G: AdamW,
    opt_D: AdamW,
) -> int:
    """
    尝试加载该 fold 的 checkpoint。
    返回：起始 epoch（若没找到 checkpoint，返回 1）
    """
    ckpt_path = OUT_DIR / f"checkpoint_fold{fold_id + 1}.pt"
    if not ckpt_path.is_file():
        return 1

    print(f"[Fold {fold_id + 1}] 发现 checkpoint，尝试从中恢复: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    disc.load_state_dict(ckpt["disc_state_dict"])
    opt_G.load_state_dict(ckpt["opt_G_state_dict"])
    opt_D.load_state_dict(ckpt["opt_D_state_dict"])

    last_epoch = int(ckpt.get("epoch", 0))
    print(f"[Fold {fold_id + 1}] 从 epoch {last_epoch} 之后继续训练")
    return last_epoch + 1


# ========= 单个预测 vs 单个真实分子的 Kabsch RMSD =========
def _kabsch_rmsd(
    pred_coord: torch.Tensor,
    gt_coord: torch.Tensor,
    gt_pad: torch.Tensor,
) -> float:
    """
    计算单个预测分子与单个真实分子的 RMSD（Å），对齐平移 + 旋转。

      pred_coord: (N,3)
      gt_coord  : (N,3)
      gt_pad    : (N,) bool, True=PAD（该位在真实分子中不存在）

    只使用 gt_pad=False 的位置进行对齐和 RMSD 计算。
    若没有任何非 PAD 原子，则返回 inf。
    """
    assert pred_coord.shape == gt_coord.shape
    assert gt_coord.ndim == 2 and gt_coord.shape[1] == 3

    mask = ~gt_pad.bool()
    idx = mask.nonzero(as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return float("inf")

    P = pred_coord[idx].double()
    Q = gt_coord[idx].double()

    # 去中心化
    P_mean = P.mean(dim=0, keepdim=True)
    Q_mean = Q.mean(dim=0, keepdim=True)
    P_centered = P - P_mean
    Q_centered = Q - Q_mean

    # Kabsch: P_centered * R ≈ Q_centered
    H = P_centered.t().mm(Q_centered)  # (3,3)
    U, S, Vh = torch.linalg.svd(H)
    R = Vh.t().mm(U.t())
    if torch.det(R) < 0:
        Vh[-1, :] *= -1.0
        R = Vh.t().mm(U.t())
    P_rot = P_centered.mm(R)

    diff = P_rot - Q_centered
    rmsd = torch.sqrt((diff * diff).sum() / P.shape[0])
    return float(rmsd)


# ========= 基于“采样 3D 结构 + RMSD 命中”的 top-k 评估 =========
def _eval_sampling_for_fold(
    model: MetabolicFinetuneModel,
    ds_all: MetabolicDataset,
    eval_rids: Set[str],
    rid_to_indices: Dict[str, List[int]],
    reaction_to_gt: Dict[str, List[str]],
    topk_list: List[int],
    fold_id: int,
    tag: str,
) -> Dict[int, Dict[str, float]]:
    """
    使用“多次采样 z”的方式评估当前 fold：

      对于每个 Reaction_ID：
        - 收集该反应所有真实产物的 3D 结构 (xyz_p, pad_p)；
        - 用第一条样本的 (底物, 酶) 构造条件，调用 model.sample_with_noise
          生成 K = max(topk_list) 个预测 3D 产物；
        - 对所有 (pred_i, gt_j) 计算 Kabsch RMSD；
        - 把 RMSD <= RMSD_DEFAULT_THRESH 视为命中，做一对一贪心匹配；
        - 针对每个 k ∈ topk_list，统计：
            * 至少命中 1 个真实代谢物的反应比例；
            * 至少命中一半真实代谢物的反应比例；
            * 全部真实代谢物都命中的反应比例；
            * 总命中真实代谢物数 / 总真实代谢物数；
            * 按反应平均的 Precision(k) / Recall(k)。

    完全只在 “当前反应的真实 3D 结构 vs 当前反应 top-k 预测 3D 结构” 上评估，
    不依赖全局候选池，也不使用指纹 Tanimoto。
    """
    model.eval()
    max_k = max(topk_list)
    rmsd_thr = RMSD_DEFAULT_THRESH

    # 各 k 的累积统计
    n_valid = 0
    total_gt_all = 0

    at_least_one = {k: 0 for k in topk_list}
    at_least_half = {k: 0 for k in topk_list}
    all_cover = {k: 0 for k in topk_list}
    total_correct = {k: 0 for k in topk_list}
    sum_precisions = {k: 0.0 for k in topk_list}
    sum_recalls = {k: 0.0 for k in topk_list}

    # 每个反应输出一行 CSV：Reaction_ID / 真实代谢物 SMILES / 各 k 的 hits 数
    rows_for_csv: List[Dict[str, str]] = []

    eval_rids_sorted = sorted(list(eval_rids))
    desc = f"[Fold {fold_id + 1}] {tag} Eval (z-sampling)"
    with torch.no_grad():
        for rid_str in tqdm(eval_rids_sorted, desc=desc, leave=False):
            idx_list = rid_to_indices.get(rid_str, None)
            if not idx_list:
                continue

            gt_smiles = reaction_to_gt.get(rid_str, None)
            if gt_smiles is None:
                gt_smiles = []

            # === 1) 收集该反应所有真实产物的 3D 结构 ===
            gt_structs = []
            for idx in idx_list:
                # 用 collate_fn 把这一条样本转成图特征
                sample = ds_all[idx]
                batch_gt = collate_fn([sample], PAD_ID, MAX_ATOMS)
                xyz_p = batch_gt["xyz_p"][0].to(DEVICE)      # (N,3)
                pad_p = batch_gt["pad_p"][0].to(DEVICE)      # (N,)
                gt_structs.append((xyz_p, pad_p))

            if len(gt_structs) == 0:
                continue

            n_gt = len(gt_structs)
            n_valid += 1
            total_gt_all += n_gt

            # === 2) 用第一条样本构造 (底物, 酶) 条件，采样 K 个预测 3D 产物 ===
            base_sample = ds_all[idx_list[0]]
            batch_one = collate_fn([base_sample], PAD_ID, MAX_ATOMS)
            for key in [
                "atom_s", "xyz_s", "pad_s",
                "atom_p", "xyz_p", "pad_p",
                "enzyme", "enzyme_mask",
            ]:
                batch_one[key] = batch_one[key].to(DEVICE)

            start_t = time.time()
            out = model.sample_with_noise(batch_one, num_samples=max_k)
            elapsed = time.time() - start_t
            if elapsed > MAX_BATCH_SECONDS:
                print(
                    f"[Fold {fold_id + 1}] {tag} Reaction_ID={rid_str} "
                    f"sample_with_noise 用时 {elapsed:.1f}s > {MAX_BATCH_SECONDS:.0f}s, "
                    f"跳过该反应的评估"
                )
                continue

            pred_coords = out["pred_coords"][0]  # (K,N,3)
            n_pred = pred_coords.shape[0]

            # === 3) 预计算所有 (gt_j, pred_i) 的 RMSD 矩阵 ===
            rmsd_mat = torch.empty(len(gt_structs), n_pred,
                                   dtype=torch.float32, device=DEVICE)
            for j, (xyz_gt, pad_gt) in enumerate(gt_structs):
                for i in range(n_pred):
                    rmsd_val = _kabsch_rmsd(pred_coords[i], xyz_gt, pad_gt)
                    rmsd_mat[j, i] = rmsd_val

            # === 4) 针对每个 k，做“一对一贪心匹配”，得到该反应在该 k 下命中的 GT 数 ===
            hits_per_k: Dict[int, int] = {}
            for k in topk_list:
                used_preds = set()
                n_hit = 0
                for j in range(len(gt_structs)):
                    best_i = None
                    best_r = float("inf")
                    for i in range(min(k, n_pred)):
                        if i in used_preds:
                            continue
                        r_ji = float(rmsd_mat[j, i])
                        if r_ji < best_r:
                            best_r = r_ji
                            best_i = i
                    if best_i is not None and best_r <= rmsd_thr:
                        n_hit += 1
                        used_preds.add(best_i)
                hits_per_k[k] = n_hit

            # === 5) 汇总全局统计 ===
            for k in topk_list:
                n_hit = hits_per_k[k]
                if n_hit >= 1:
                    at_least_one[k] += 1
                # 至少一半：向上取整
                if n_hit >= max(1, int(np.ceil(n_gt / 2.0))):
                    at_least_half[k] += 1
                if n_hit == n_gt:
                    all_cover[k] += 1
                total_correct[k] += n_hit

                prec_k = n_hit / float(k)
                rec_k = n_hit / float(n_gt)
                sum_precisions[k] += prec_k
                sum_recalls[k] += rec_k

            # === 6) 写一行 CSV（方便后续分析） ===
            row = {
                "Reaction_ID": rid_str,
                "Num_GT": str(n_gt),
                "GT_Metabolites": ";".join(gt_smiles),
            }
            for k in topk_list:
                row[f"Hits_Top{k}"] = str(hits_per_k[k])
            rows_for_csv.append(row)

    # === 输出 CSV ===
    if rows_for_csv:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        df_pred = pd.DataFrame(rows_for_csv)
        csv_name = f"predictions_fold{fold_id + 1}_{tag}_zsamples.csv"
        csv_path = PRED_DIR / csv_name
        df_pred.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[Fold {fold_id + 1}] {tag} 采样式预测结果已保存到: {csv_path}")

    if n_valid == 0 or total_gt_all == 0:
        print(f"[Fold {fold_id + 1}] {tag} 没有可评估样本。")
        return {}

    # === 把累计量转换成百分比指标 ===
    results: Dict[int, Dict[str, float]] = {}
    for k in topk_list:
        metrics_k = {
            "AtLeastOne": 100.0 * at_least_one[k] / n_valid,
            "AtLeastHalf": 100.0 * at_least_half[k] / n_valid,
            "All": 100.0 * all_cover[k] / n_valid,
            "TotalIdentified": 100.0 * total_correct[k] / total_gt_all,
            "Precision": 100.0 * (sum_precisions[k] / n_valid),
            "Recall": 100.0 * (sum_recalls[k] / n_valid),
        }
        results[k] = metrics_k
        print(f"[Fold {fold_id + 1}] {tag} Top-{k} 指标 (单位: %)：")
        print(
            f"  At least one metabolite    : {metrics_k['AtLeastOne']:.4f}"
            f"\n  At least half metabolite    : {metrics_k['AtLeastHalf']:.4f}"
            f"\n  All metabolites             : {metrics_k['All']:.4f}"
            f"\n  Total identified metabolites: {metrics_k['TotalIdentified']:.4f}"
            f"\n  Precision                   : {metrics_k['Precision']:.4f}"
            f"\n  Recall                      : {metrics_k['Recall']:.4f}"
        )

    return results


# ========= 主训练 + 5 折交叉验证 =========
def run_kfold_finetune(num_folds: int = 5,
                       topk_list: List[int] | None = None):
    if topk_list is None:
        # 默认为 MetaTrans 风格：Top-5,10,12,15
        topk_list = [5, 10, 12, 15]

    print(f"[Config] device={DEVICE}, batch_size={BATCH_SIZE}, epochs={NUM_EPOCHS}")
    print(f"[Data ] CSV: {CSV_PATH}")
    print(f"[Shared] Encoder: {ENC_PATH}")
    print(f"[Shared] Decoder: {DEC_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # 固定随机种子（初始）
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # 1) 完整 dataset（已按 (Reaction_ID, product) 展开）
    ds_all = MetabolicDataset(CSV_PATH, split="train", eval_ratio=0.0, random_state=SEED)
    n_samples = len(ds_all)
    print(f"[Data] 展开后的样本数(Reaction_ID, product 对): {n_samples}")

    # 记录每个样本的 Reaction_ID
    reaction_ids = [str(it["reaction_id"]) for it in ds_all.items]
    unique_rids = sorted(set(reaction_ids))
    n_reactions = len(unique_rids)
    print(f"[Data] 反应数 (不同 Reaction_ID): {n_reactions}")

    # 2) 构建 Reaction_ID -> 真实产物 SMILES（仅用于 CSV 输出）
    reaction_to_gt = _build_reaction_gt(CSV_PATH)
    print(f"[Eval] 具有真实产物 SMILES 的反应数: {len(reaction_to_gt)}")

    # 3) Reaction_ID -> 所有样本索引（用于评估时收集 3D GT）
    rid_to_indices: Dict[str, List[int]] = {}
    for idx, rid in enumerate(reaction_ids):
        rid_to_indices.setdefault(str(rid), []).append(idx)

    # 4) Reaction_ID 级别的 K 折划分
    rng = np.random.default_rng(SEED)
    rid_perm = rng.permutation(unique_rids)
    rid_folds = np.array_split(rid_perm, num_folds)

    fold_results: Dict[int, List[Dict[str, float]]] = {k: [] for k in topk_list}
    bce_logits = F.binary_cross_entropy_with_logits

    # 5) 逐 fold 训练 + 评估
    for fold_id in range(num_folds):
        print("\n" + "=" * 80)
        print(f"[Fold {fold_id + 1}/{num_folds}] 开始训练 & 评估")
        print("=" * 80)

        fold_model_path = OUT_DIR / f"finetune_fold{fold_id + 1}.pt"
        fold_ckpt_path = OUT_DIR / f"checkpoint_fold{fold_id + 1}.pt"

        eval_rids = set(str(r) for r in rid_folds[fold_id])
        train_rids = set(r for r in unique_rids if r not in eval_rids)

        train_idx = [i for i, rid in enumerate(reaction_ids) if rid in train_rids]
        eval_idx = [i for i, rid in enumerate(reaction_ids) if rid in eval_rids]

        print(f"[Fold {fold_id + 1}] 训练样本数: {len(train_idx)}, 验证样本数: {len(eval_idx)}")
        print(f"[Fold {fold_id + 1}] 训练反应数: {len(train_rids)}, 验证反应数: {len(eval_rids)}")

        # DataLoader（训练）
        train_subset = Subset(ds_all, train_idx)
        dl_train = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=lambda b: collate_fn(b, PAD_ID, MAX_ATOMS),
            pin_memory=True,
        )

        # DataLoader（验证，用于早停）
        eval_subset = Subset(ds_all, eval_idx)
        dl_eval = DataLoader(
            eval_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=lambda b: collate_fn(b, PAD_ID, MAX_ATOMS),
            pin_memory=True,
        )

        # 初始化模型 & 判别器 & 优化器
        model = MetabolicFinetuneModel(noise_std=NOISE_STD).to(DEVICE)
        disc = MultiTaskDiscriminator(hidden=model.d_model).to(DEVICE)

        # -------- 生成器分组学习率：encoder/decoder 用更小 LR，新加模块用主 LR --------
        enc_dec_params: List[torch.nn.Parameter] = []
        new_params: List[torch.nn.Parameter] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if ("encoder" in name) or ("decoder" in name):
                enc_dec_params.append(p)
            else:
                new_params.append(p)

        opt_G = AdamW(
            [
                {"params": enc_dec_params, "lr": LR * 0.1},  # 预训练部分：小 LR
                {"params": new_params,    "lr": LR},         # 新加部分：正常 LR
            ],
            weight_decay=WEIGHT_DECAY,
        )
        opt_D = AdamW(disc.parameters(), lr=D_LR, weight_decay=WEIGHT_DECAY)

        # 训练部分：支持三种情况
        if fold_model_path.is_file():
            # 已经训练完这个 fold：直接加载最终模型，跳过训练
            print(f"[Fold {fold_id + 1}] 检测到已存在最终模型，跳过训练，加载: {fold_model_path}")
            ck = torch.load(str(fold_model_path), map_location=DEVICE, weights_only=False)
            model.load_state_dict(ck["model_state_dict"])
        else:
            # 若没有最终模型，看是否有中间 checkpoint
            if not fold_ckpt_path.is_file():
                _load_ckpt_or_die(model)
                print(f"[Fold {fold_id + 1}] 无 checkpoint，从 epoch 1 开始训练")
                start_epoch = 1
            else:
                # 有 checkpoint，从 checkpoint 恢复
                start_epoch = _try_load_checkpoint(
                    fold_id=fold_id,
                    model=model,
                    disc=disc,
                    opt_G=opt_G,
                    opt_D=opt_D,
                )

            best_val_loss = float("inf")
            best_epoch = 0
            patience_counter = 0
            best_model_state = None  # type: ignore

            running_sup = 0.0
            running_g_total = 0.0

            for ep in range(start_epoch, NUM_EPOCHS + 1):
                model.train()
                disc.train()
                running_sup = 0.0
                running_g_total = 0.0

                # 本 epoch 是否启用 GAN（前若干个 epoch 只做监督重建）
                use_gan_this_epoch = (ep > GAN_WARMUP_EPOCHS)

                pbar = tqdm(dl_train, ncols=100,
                            desc=f"[Fold {fold_id + 1}] Train ep {ep}/{NUM_EPOCHS}")
                for it, batch in enumerate(pbar, 1):
                    batch_start_time = time.time()

                    for k in [
                        "atom_s", "xyz_s", "pad_s",
                        "atom_p", "xyz_p", "pad_p",
                        "enzyme", "enzyme_mask",
                    ]:
                        batch[k] = batch[k].to(DEVICE, non_blocking=True)

                    B = batch["atom_s"].size(0)
                    batch_rids = batch.get("reaction_id", None)

                    use_gan_batch = use_gan_this_epoch and (B >= 2)

                    if not use_gan_batch:
                        # batch 太小或在 warmup 期：只做监督重建
                        loss_sup, ld, _ = model(batch, need_repr_for_gan=False)
                        loss_sup = torch.nan_to_num(loss_sup, nan=0.0, posinf=1e4, neginf=1e4)
                        if not torch.isfinite(loss_sup):
                            print(f"[Fold {fold_id + 1}] ep {ep} it {it}: loss_sup 非有限, 跳过该 batch")
                            continue

                        elapsed = time.time() - batch_start_time
                        if elapsed > MAX_BATCH_SECONDS:
                            msg = (
                                f"[Fold {fold_id + 1}] ep {ep} it {it}: "
                                f"单个 batch 用时 {elapsed:.1f}s > {MAX_BATCH_SECONDS:.0f}s, "
                                f"跳过该 batch 的更新"
                            )
                            if batch_rids is not None:
                                msg += f"，reaction_id={_format_rids_for_log(batch_rids)}"
                            print(msg)
                            continue

                        opt_G.zero_grad(set_to_none=True)
                        loss_sup.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt_G.step()

                        running_sup += float(ld["loss"].item())
                        running_g_total += float(ld["loss"].item())
                        avg_sup = running_sup / it
                        avg_g = running_g_total / it
                        pbar.set_postfix(
                            sup=f"{avg_sup:.4f}",
                            g_tot=f"{avg_g:.4f}",
                            atom=f"{float(ld['loss_atom']):.4f}",
                            bond=f"{float(ld['loss_bond']):.4f}",
                        )
                        continue

                    # ---- 判别器更新 ----
                    for _ in range(D_STEPS):
                        with torch.no_grad():
                            _, _, aux = model(batch, need_repr_for_gan=True)
                        sub_vec = aux["sub_vec"]
                        prod_fake_vec = aux["prod_fake_vec"]
                        prod_real_vec = aux["prod_real_vec"]
                        enz_vec = aux["enz_vec"]

                        sub_neg = sub_vec.roll(shifts=1, dims=0)
                        enz_neg = enz_vec.roll(shifts=1, dims=0)
                        prod_neg = prod_real_vec.roll(shifts=1, dims=0)

                        ones = torch.ones(B, device=DEVICE)
                        zeros = torch.zeros(B, device=DEVICE)

                        # 三元组一致性判别器
                        logits_t1_pos = disc.tri1_logits(sub_vec, enz_vec, prod_real_vec)
                        logits_t1_pos = torch.nan_to_num(logits_t1_pos, nan=0.0, posinf=10.0, neginf=-10.0)
                        logits_t1_neg = disc.tri1_logits(sub_neg, enz_vec, prod_real_vec)
                        logits_t1_neg = torch.nan_to_num(logits_t1_neg, nan=0.0, posinf=10.0, neginf=-10.0)

                        logits_t2_pos = disc.tri2_logits(sub_vec, enz_vec, prod_real_vec)
                        logits_t2_pos = torch.nan_to_num(logits_t2_pos, nan=0.0, posinf=10.0, neginf=-10.0)
                        logits_t2_neg = disc.tri2_logits(sub_vec, enz_neg, prod_real_vec)
                        logits_t2_neg = torch.nan_to_num(logits_t2_neg, nan=0.0, posinf=10.0, neginf=-10.0)

                        logits_t3_pos = disc.tri3_logits(sub_vec, enz_vec, prod_real_vec)
                        logits_t3_pos = torch.nan_to_num(logits_t3_pos, nan=0.0, posinf=10.0, neginf=-10.0)
                        logits_t3_neg = disc.tri3_logits(sub_vec, enz_vec, prod_neg)
                        logits_t3_neg = torch.nan_to_num(logits_t3_neg, nan=0.0, posinf=10.0, neginf=-10.0)

                        loss_t1 = 0.5 * (bce_logits(logits_t1_pos, ones) + bce_logits(logits_t1_neg, zeros))
                        loss_t2 = 0.5 * (bce_logits(logits_t2_pos, ones) + bce_logits(logits_t2_neg, zeros))
                        loss_t3 = 0.5 * (bce_logits(logits_t3_pos, ones) + bce_logits(logits_t3_neg, zeros))
                        loss_tri = (loss_t1 + loss_t2 + loss_t3) / 3.0

                        # 对抗判别器
                        logits_adv_real = disc.adv_logits(prod_real_vec)
                        logits_adv_real = torch.nan_to_num(logits_adv_real, nan=0.0, posinf=10.0, neginf=-10.0)
                        logits_adv_fake = disc.adv_logits(prod_fake_vec.detach())
                        logits_adv_fake = torch.nan_to_num(logits_adv_fake, nan=0.0, posinf=10.0, neginf=-10.0)

                        loss_adv_D = 0.5 * (bce_logits(logits_adv_real, ones) +
                                            bce_logits(logits_adv_fake, zeros))

                        loss_D = loss_tri + loss_adv_D
                        loss_D = torch.nan_to_num(loss_D, nan=0.0, posinf=1e4, neginf=1e4)
                        if not torch.isfinite(loss_D):
                            print(f"[Fold {fold_id + 1}] ep {ep} it {it}: loss_D 非有限, 跳过 D-step")
                            continue

                        elapsed_d = time.time() - batch_start_time
                        if elapsed_d > MAX_BATCH_SECONDS:
                            msg = (
                                f"[Fold {fold_id + 1}] ep {ep} it {it}: "
                                f"D-step 用时 {elapsed_d:.1f}s > {MAX_BATCH_SECONDS:.0f}s, "
                                f"跳过 D-step 更新"
                            )
                            if batch_rids is not None:
                                msg += f"，reaction_id={_format_rids_for_log(batch_rids)}"
                            print(msg)
                            continue

                        opt_D.zero_grad(set_to_none=True)
                        loss_D.backward()
                        torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                        opt_D.step()

                    # ---- 生成器更新 ----
                    loss_sup, ld, aux = model(batch, need_repr_for_gan=True)
                    loss_sup = torch.nan_to_num(loss_sup, nan=0.0, posinf=1e4, neginf=1e4)
                    if not torch.isfinite(loss_sup):
                        print(f"[Fold {fold_id + 1}] ep {ep} it {it}: loss_sup 非有限, 跳过 G-step")
                        continue

                    sub_vec = aux["sub_vec"]
                    prod_fake_vec = aux["prod_fake_vec"]
                    enz_vec = aux["enz_vec"]

                    ones = torch.ones(B, device=DEVICE)

                    logits_adv_fake = disc.adv_logits(prod_fake_vec)
                    logits_adv_fake = torch.nan_to_num(logits_adv_fake, nan=0.0, posinf=10.0, neginf=-10.0)
                    loss_g_adv = bce_logits(logits_adv_fake, ones)

                    logits_t3_fake = disc.tri3_logits(sub_vec, enz_vec, prod_fake_vec)
                    logits_t3_fake = torch.nan_to_num(logits_t3_fake, nan=0.0, posinf=10.0, neginf=-10.0)
                    loss_g_tri = bce_logits(logits_t3_fake, ones)

                    loss_G = loss_sup + LAMBDA_ADV * loss_g_adv + LAMBDA_TRI * loss_g_tri
                    loss_G = torch.nan_to_num(loss_G, nan=0.0, posinf=1e4, neginf=1e4)
                    if not torch.isfinite(loss_G):
                        print(f"[Fold {fold_id + 1}] ep {ep} it {it}: loss_G 非有限, 跳过 G-step")
                        continue

                    elapsed = time.time() - batch_start_time
                    if elapsed > MAX_BATCH_SECONDS:
                        msg = (
                            f"[Fold {fold_id + 1}] ep {ep} it {it}: "
                            f"单个 batch 用时 {elapsed:.1f}s > {MAX_BATCH_SECONDS:.0f}s, "
                            f"跳过该 batch 的 G-step 更新"
                        )
                        if batch_rids is not None:
                            msg += f"，reaction_id={_format_rids_for_log(batch_rids)}"
                        print(msg)
                        continue

                    opt_G.zero_grad(set_to_none=True)
                    loss_G.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt_G.step()

                    running_sup += float(ld["loss"].item())
                    running_g_total += float(loss_G.item())
                    avg_sup = running_sup / it
                    avg_g = running_g_total / it
                    pbar.set_postfix(
                        sup=f"{avg_sup:.4f}",
                        g_tot=f"{avg_g:.4f}",
                        atom=f"{float(ld['loss_atom']):.4f}",
                        bond=f"{float(ld['loss_bond']):.4f}",
                    )

                # 每个 epoch 结束保存一次 checkpoint
                _save_checkpoint(
                    fold_id=fold_id,
                    epoch=ep,
                    model=model,
                    disc=disc,
                    opt_G=opt_G,
                    opt_D=opt_D,
                )

                # ---------- 验证集重建损失 (用于 Early Stopping) ----------
                model.eval()
                val_loss_sum = 0.0
                val_batches = 0
                with torch.no_grad():
                    pbar_val = tqdm(dl_eval, ncols=100,
                                    desc=f"[Fold {fold_id + 1}] Val   ep {ep}/{NUM_EPOCHS}",
                                    leave=False)
                    for it_val, batch in enumerate(pbar_val, 1):
                        batch_start_time = time.time()

                        for k in [
                            "atom_s", "xyz_s", "pad_s",
                            "atom_p", "xyz_p", "pad_p",
                            "enzyme", "enzyme_mask",
                        ]:
                            batch[k] = batch[k].to(DEVICE, non_blocking=True)

                        batch_rids = batch.get("reaction_id", None)

                        loss_sup_val, ld_val, _ = model(batch, need_repr_for_gan=False)
                        loss_sup_val = torch.nan_to_num(loss_sup_val, nan=0.0, posinf=1e4, neginf=1e4)
                        if not torch.isfinite(loss_sup_val):
                            continue

                        elapsed_val = time.time() - batch_start_time
                        if elapsed_val > MAX_BATCH_SECONDS:
                            msg = (
                                f"[Fold {fold_id + 1}] ep {ep} val-batch {it_val}: "
                                f"用时 {elapsed_val:.1f}s > {MAX_BATCH_SECONDS:.0f}s, "
                                f"跳过该 val batch"
                            )
                            if batch_rids is not None:
                                msg += f"，reaction_id={_format_rids_for_log(batch_rids)}"
                            print(msg)
                            continue

                        val_loss_sum += float(ld_val["loss"].item())
                        val_batches += 1

                val_loss = val_loss_sum / max(val_batches, 1)
                print(f"[Fold {fold_id + 1}] ep {ep}: val_sup={val_loss:.4f} (best={best_val_loss:.4f})")

                # 更新早停状态（仍然基于监督重建损失）
                if val_loss + EARLY_STOPPING_DELTA < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = ep
                    patience_counter = 0
                    best_model_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    print(f"[Fold {fold_id + 1}] ep {ep}: val 未提升, patience={patience_counter}/{EARLY_STOPPING_PATIENCE}")
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        print(f"[Fold {fold_id + 1}] 触发早停, 在 epoch {best_epoch} 取得最优 val_sup={best_val_loss:.4f}")
                        break

            # 所有 epoch 完成或早停后，加载最佳模型权重并保存最终模型
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            else:
                print(f"[Fold {fold_id + 1}] 警告: 未记录 best_model_state, 使用最后一个 epoch 的权重。")

            torch.save({"model_state_dict": model.state_dict()}, fold_model_path)
            print(f"[Fold {fold_id + 1}] 已保存最终模型到: {fold_model_path}")
            if fold_ckpt_path.is_file():
                os.remove(fold_ckpt_path)
                print(f"[Fold {fold_id + 1}] 已删除 checkpoint: {fold_ckpt_path}")

        # ---------- 采样式 3D-RMSD top-k 评估 ---------- #
        final_metrics_per_k = _eval_sampling_for_fold(
            model, ds_all, eval_rids, rid_to_indices,
            reaction_to_gt, topk_list, fold_id, tag="FINAL"
        )

        for k in topk_list:
            if k in final_metrics_per_k:
                fold_results[k].append(final_metrics_per_k[k])

    # 6) 汇总 K 折结果：均值 ± 标准差
    metric_names = [
        "AtLeastOne",
        "AtLeastHalf",
        "All",
        "TotalIdentified",
        "Precision",
        "Recall",
    ]
    metric_labels = {
        "AtLeastOne": "At least one metabolite (%)",
        "AtLeastHalf": "At least half metabolite (%)",
        "All": "All metabolites (%)",
        "TotalIdentified": "Total identified metabolites (%)",
        "Precision": "Precision (%)",
        "Recall": "Recall (%)",
    }

    table_rows = []
    for m in metric_names:
        row = {}
        for k in topk_list:
            vals = [fr[m] for fr in fold_results[k]]
            if len(vals) == 0:
                cell = "nan±nan"
            else:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                cell = f"{mean:.4f}±{std:.4f}"
            row[f"Top-{k}"] = cell
        table_rows.append(row)

    df = pd.DataFrame(table_rows,
                      index=[metric_labels[m] for m in metric_names])

    print("\n" + "=" * 80)
    print("5 折交叉验证结果 (均值 ± 标准差, 单位: %)")
    print("=" * 80)
    print(df.to_string())

    csv_path = OUT_DIR / "kfold_metrics.csv"
    df.to_csv(csv_path, encoding="utf-8-sig")
    print(f"\n[Done] 已将 5 折交叉验证结果保存到: {csv_path}")


if __name__ == "__main__":
    run_kfold_finetune()
