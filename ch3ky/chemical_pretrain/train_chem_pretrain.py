# -*- coding: utf-8 -*-
# train_stage2_v2.py
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from rdkit import Chem  # 用于生成完整的元素周期表映射

from chemical_pretrain.reaction_dataset import ReactionDataset, reaction_collate
from chem_reaction_model import ChemicalReactionModelV2  # 引用新模型
from chemical_pretrain.utils_geometry import kabsch_rotate
from chemical_pretrain.utils import set_seed, ensure_dir
from chemical_pretrain.constants import VOCAB_SIZE, PAD_ID, MASK_ID

# === 配置 ===
CONFIG = {
    "train_txt": "data/train_10.txt",
    "cache_path": "data/cached_reaction.pt",
    "pretrain_weights": "shared_encoder.pt",
    "out_dir": "runs_stage2_v2",
    "sample_dir": "runs_stage2_v2/samples",

    "epochs": 50,
    "batch_size": 16,
    "lr": 2e-4,
    "log_interval": 50,
    "eval_interval": 1,
}


# ========================================================================
# 1. 工具函数：ID ↔ 元素符号映射 (完整版)
# ========================================================================
def build_id2atom():
    """
    根据 VOCAB_SIZE=120 的约定，构造完整的 id->元素符号 映射表：
        0: PAD
        1: MASK
        2..119: 原子序数 1..118 对应的元素
    """
    pt = Chem.GetPeriodicTable()
    id2atom = {
        PAD_ID: "PAD",
        MASK_ID: "MASK",
    }

    # 遍历 1 到 118 号元素
    for idx in range(2, VOCAB_SIZE):
        z = idx - 1  # 约定：ID = Z + 1
        try:
            symbol = pt.GetElementSymbol(int(z))
        except Exception:
            symbol = f"X{z}"
        id2atom[idx] = symbol
    return id2atom


# ========================================================================
# 2. 类别权重计算 (针对不平衡数据) - 修复版
# ========================================================================
def compute_class_weights(dataset, device):
    """
    统计数据集中原子出现的频率，计算反向权重。
    让稀有原子权重更高，C/H 权重更低。

    [修复] 使用 DataLoader 读取，确保 keys (p_atom_ids) 存在且正确。
    """
    print("Computing class weights (scanning dataset via DataLoader)...")
    counts = torch.zeros(VOCAB_SIZE)

    # 创建一个临时的 loader，使用 collate_fn 确保数据格式正确
    # num_workers=4 加速读取
    temp_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=reaction_collate,
        num_workers=4
    )

    # 只需要扫描一部分数据即可 (例如 200 个 batch，约 12800 条数据)
    max_batches = 200
    scanned_batches = 0

    for batch in temp_loader:
        if batch is None: continue

        # 此时 batch 经过了 collate，肯定包含 'p_atom_ids'
        if 'p_atom_ids' in batch:
            p_ids = batch['p_atom_ids']  # (B, T)

            # Flatten
            flat_ids = p_ids.reshape(-1)

            # 过滤掉 PAD
            valid_ids = flat_ids[flat_ids != PAD_ID]

            # 统计当前 batch 的词频
            # bincount 速度很快，但要求输入非负整数
            # 确保在 CPU 上操作，避免显存占用
            if valid_ids.numel() > 0:
                batch_counts = torch.bincount(valid_ids.cpu(), minlength=VOCAB_SIZE)
                # 累加到总 counts (截断到 VOCAB_SIZE)
                counts[:len(batch_counts)] += batch_counts.float()

        scanned_batches += 1
        if scanned_batches >= max_batches:
            break

    print(f"Scanned {scanned_batches} batches.")

    # 防止除以0
    counts = torch.clamp(counts, min=1.0)

    # 策略：Sqrt Inverse Frequency (平方根倒数，比直接倒数更平滑)
    weights = 1.0 / torch.sqrt(counts)

    # Normalize: 让平均权重为 1
    weights = weights / weights.mean()

    # 手动调整特殊 token
    weights[0] = 0.0  # PAD
    weights[1] = 0.1  # MASK

    # 打印部分权重供检查
    id2atom = build_id2atom()
    print("Class Weights Top 5 Lowest (Most Frequent):")
    vals, idxs = torch.topk(weights, 5, largest=False)
    for v, idx in zip(vals, idxs):
        if idx.item() in id2atom:
            print(f"  {id2atom[idx.item()]}: {v:.4f}")

    print("Class Weights Top 5 Highest (Rare):")
    vals, idxs = torch.topk(weights, 5, largest=True)
    for v, idx in zip(vals, idxs):
        if idx.item() in id2atom and v > 0:  # 过滤未出现的
            print(f"  {id2atom[idx.item()]}: {v:.4f}")

    return weights.to(device)


# ========================================================================
# 3. 评估与可视化函数
# ========================================================================
def calculate_metrics(logits_atom, pred_coords, tgt_ids, tgt_coords, tgt_mask):
    """
    计算指标:
        - Atom Accuracy
        - RMSD (Root Mean Square Deviation)
    """
    with torch.no_grad():
        # 1. Atom Accuracy
        pred_ids = logits_atom.argmax(dim=-1)
        valid_mask = ~tgt_mask  # True = 有效原子
        correct = (pred_ids == tgt_ids) & valid_mask
        num_valid = valid_mask.sum().float()

        acc = 0.0
        if num_valid > 0:
            acc = (correct.sum().float() / (num_valid + 1e-8)).item()

        # 2. RMSD
        # pred_coords 已经是对齐后的坐标
        diff = (pred_coords - tgt_coords) * valid_mask.unsqueeze(-1).float()
        per_n_valid = valid_mask.sum(dim=-1).float().clamp(min=1.0)  # (B,)

        # MSE per molecule -> RMSD per molecule -> Mean
        mse_per_mol = (diff.pow(2).sum(dim=-1).sum(dim=-1) / per_n_valid)
        rmsd = torch.sqrt(mse_per_mol).mean().item()

    return acc, rmsd


def save_xyz(path, atom_ids, coords, id2atom, comment="Generated"):
    """保存为 .xyz 格式"""
    lines = []
    # 过滤 PAD 和 MASK
    valid_indices = [
        i for i, idx in enumerate(atom_ids.cpu().tolist())
        if idx not in (PAD_ID, MASK_ID)
    ]

    lines.append(f"{len(valid_indices)}")
    lines.append(comment)

    coords = coords.cpu().numpy()
    atom_ids = atom_ids.cpu().tolist()

    for i in valid_indices:
        atom_id = atom_ids[i]
        symbol = id2atom.get(atom_id, "X")
        x, y, z = coords[i]
        lines.append(f"{symbol} {x:.4f} {y:.4f} {z:.4f}")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def evaluate_and_save(model, loader, device, epoch, cfg, id2atom):
    """
    遍历整个验证集计算指标，并保存第一个 Batch 的可视化结果
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_rmsd = 0.0
    steps = 0
    saved_sample = False

    ensure_dir(cfg['sample_dir'])

    with torch.no_grad():
        for batch in loader:
            if batch is None: continue

            r_ids = batch['r_atom_ids'].to(device)
            r_coords = batch['r_coords'].to(device)
            r_mask = batch['r_mask'].to(device)
            p_ids = batch['p_atom_ids'].to(device)
            p_coords = batch['p_coords'].to(device)
            p_mask = batch['p_mask'].to(device)

            # Forward (V2 模型返回 (logits, coords_aligned))
            loss, _, (logits_atom, coords_aligned) = model(
                r_ids, r_coords, r_mask, p_ids, p_coords, p_mask
            )

            # 截断到真实长度 T 以计算指标
            T = min(model.max_prod_len, p_ids.shape[1])
            logits_slice = logits_atom[:, :T, :]
            coords_slice = coords_aligned[:, :T, :]  # 已经过 Kabsch 对齐
            p_ids_slice = p_ids[:, :T]
            p_coords_slice = p_coords[:, :T, :]
            p_mask_slice = p_mask[:, :T]

            # 计算指标
            acc, rmsd = calculate_metrics(
                logits_slice, coords_slice,
                p_ids_slice, p_coords_slice, p_mask_slice
            )

            total_loss += loss.item()
            total_acc += acc
            total_rmsd += rmsd
            steps += 1

            # 保存第一个 Batch 的第一个样本用于可视化
            if not saved_sample:
                idx = 0

                # 1. 真实产物
                save_xyz(
                    os.path.join(cfg['sample_dir'], f"ep{epoch:03d}_real.xyz"),
                    p_ids_slice[idx], p_coords_slice[idx], id2atom,
                    comment=f"Epoch {epoch} Real Product"
                )

                # 2. 预测产物 (使用对齐后的坐标)
                pred_ids = logits_slice.argmax(dim=-1)
                save_xyz(
                    os.path.join(cfg['sample_dir'], f"ep{epoch:03d}_pred.xyz"),
                    pred_ids[idx], coords_slice[idx], id2atom,
                    comment=f"Epoch {epoch} Predicted Product (RMSD={rmsd:.2f})"
                )

                saved_sample = True

    if steps == 0:
        return 0.0, 0.0, 0.0

    return total_loss / steps, total_acc / steps, total_rmsd / steps


# ========================================================================
# 主训练循环
# ========================================================================
def main():
    cfg = CONFIG
    set_seed(2024)
    ensure_dir(cfg['out_dir'])
    ensure_dir(cfg['sample_dir'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Dataset
    # 注意：ReactionDataset 加载可能需要一点时间
    full_ds = ReactionDataset(cfg['train_txt'], cfg['cache_path'])
    n_train = int(0.9 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [n_train, len(full_ds) - n_train])

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, collate_fn=reaction_collate,
                              num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, collate_fn=reaction_collate,
                            num_workers=2)

    # 2. Compute Class Weights (使用修正后的函数)
    class_weights = compute_class_weights(full_ds, device)

    # 3. Model V2
    model = ChemicalReactionModelV2(cfg['pretrain_weights'], class_weights=class_weights).to(device)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=cfg['lr'], steps_per_epoch=len(train_loader), epochs=cfg['epochs'])

    # 5. Build ID -> Atom Mapping (用于保存 .xyz)
    id2atom = build_id2atom()

    print("Start Training V2...")
    best_val_rmsd = float("inf")

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total_loss = 0
        steps = 0

        # 动态权重策略
        if epoch < 5:
            coord_w = 0.5
        else:
            coord_w = 2.0

        for batch in train_loader:
            if batch is None: continue

            r_ids = batch['r_atom_ids'].to(device)
            r_coords = batch['r_coords'].to(device)
            r_mask = batch['r_mask'].to(device)
            p_ids = batch['p_atom_ids'].to(device)
            p_coords = batch['p_coords'].to(device)
            p_mask = batch['p_mask'].to(device)

            optimizer.zero_grad()

            # Forward
            loss, logs, _ = model(r_ids, r_coords, r_mask, p_ids, p_coords, p_mask, coord_loss_weight=coord_w)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            steps += 1

            if steps % cfg['log_interval'] == 0:
                print(
                    f"Ep {epoch} Step {steps} | Loss: {loss.item():.4f} | Atom: {logs['l_atom']:.4f} | Coord: {logs['l_coord']:.4f} | lr: {scheduler.get_last_lr()[0]:.6f}")

        # Eval & Save
        if epoch % cfg['eval_interval'] == 0:
            val_loss, val_acc, val_rmsd = evaluate_and_save(model, val_loader, device, epoch, cfg, id2atom)

            print(f"=== Epoch {epoch} Eval ===")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | Val RMSD: {val_rmsd:.4f}")
            print(f"Samples saved to: {cfg['sample_dir']}")

            # 保存 Best Model
            if val_rmsd < best_val_rmsd:
                best_val_rmsd = val_rmsd
                torch.save(model.state_dict(), os.path.join(cfg['out_dir'], "best_stage2.pt"))
                print(f"New Best Model Saved (RMSD: {best_val_rmsd:.4f})")

            # 保存 Latest Model
            torch.save(model.state_dict(), os.path.join(cfg['out_dir'], "latest_stage2.pt"))


if __name__ == "__main__":
    main()