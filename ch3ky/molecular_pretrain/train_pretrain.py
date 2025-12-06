# -*- coding: utf-8 -*-
import os
import logging
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split

from dataset_sdf import SDFMolDataset, batch_collate
from pretrain_model import MoleculePretrainModel
from utils import set_seed, ensure_dir, Meter

# 禁用 RDKit 所有级别的日志
logging.getLogger('rdkit').setLevel(logging.CRITICAL)

# ============ 配置 ============ #
# 如需改动，直接改这里即可；无需命令行参数
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # 数据路径：默认使用与本脚本同级目录下的 pcqm4m-v2-train/v1-train.sdf
    "sdf": os.path.join(SCRIPT_DIR, "pcqm4m-v2-train", "pcqm4m-v2-train.sdf"),
    "out": os.path.join(SCRIPT_DIR, "runs_pretrain"),

    # 数据/训练
    "max_mols": 5000000,     # 最多读取分子数
    "max_atoms": 512,        # 超过此原子数的分子将被跳过
    "batch_size": 128,
    "epochs": 50,
    "lr": 1e-4,
    "seed": 2024,

    # 模型（与论文/实现一致）
    "d_model": 256,
    "n_heads": 8,            # ★ 改为 8 个头，和模型实现/论文一致
    "n_layers_enc": 15,
    "n_layers_dec": 5,
    "d_ff": 1024,
    "rbf_k": 64,
    "rbf_mu_max": 10.0,
    "p_mask": 0.15,
    "max_len": 512,
    "dropout": 0.1,
}
# ============================ #

def main():
    cfg = CONFIG
    print("========== 预训练配置 ==========")
    for k, v in cfg.items():
        # 把全部配置打印出来更清晰；如果只想看部分可以自行筛选
        print(f"{k}: {v}")
    print("================================")

    set_seed(cfg["seed"])
    ensure_dir(cfg["out"])

    # 数据集
    sdf_path = cfg["sdf"]
    print(f"读取数据：{sdf_path}")
    ds = SDFMolDataset(
        sdf_path, max_mols=cfg["max_mols"], max_atoms=cfg["max_atoms"],
        sanitize=True, remove_hs=False
    )

    # 简单 9:1 划分
    n_total = len(ds)
    n_valid = max(1, int(0.1 * n_total))
    n_train = n_total - n_valid
    train_set, valid_set = random_split(
        ds,
        [n_train, n_valid],
        generator=torch.Generator().manual_seed(cfg["seed"])
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=batch_collate,
        num_workers=0,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=batch_collate,
        num_workers=0,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    # 词表大小：0-[PAD]，1-[MASK]，2..119 对应原子号1..118
    vocab_size = 120

    model = MoleculePretrainModel(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        n_layers_enc=cfg["n_layers_enc"],
        n_layers_dec=cfg["n_layers_dec"],
        rbf_k=cfg["rbf_k"],
        rbf_mu_max=cfg["rbf_mu_max"],
        p_mask=cfg["p_mask"],
        max_len=cfg["max_len"],
        dropout=cfg["dropout"],
        # 下面两个是坐标/距离重建的 loss 权重，
        # 按论文需求可以设成 1.0 或适当调节
        coord_loss_weight=1.0,
        dist_loss_weight=1.0,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=max(1, cfg["epochs"])
    )

    best_val = 1e9
    no_improve = 0  # early stopping counter
    patience = 5    # early stopping patience (epochs)

    # 检查是否有保存的checkpoint
    checkpoint_path = os.path.join(cfg["out"], "checkpoint_latest.pt")
    if os.path.exists(checkpoint_path):
        print("加载上次训练的模型...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_start = checkpoint["epoch"]
        best_val = checkpoint["best_val"]
        print(f"从 epoch {epoch_start} 继续训练。")
    else:
        epoch_start = 0
        print("没有找到已有的 checkpoint，重新开始训练。")

    for ep in range(epoch_start + 1, cfg["epochs"] + 1):
        # -------- 训练 --------
        model.train()
        meter = Meter()
        pbar = tqdm(
            train_loader,
            ncols=100,
            desc=f"Epoch {ep}/{cfg['epochs']} [train]",
            dynamic_ncols=True,
        )
        for batch in pbar:
            atom_ids = batch["atom_ids"].to(device)
            coords = batch["coords"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            optim.zero_grad(set_to_none=True)
            loss, loss_dict, _ = model(atom_ids, coords, pad_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optim.step()

            meter.update(loss_dict)
            # 这里的 loss_dict 现在只有：
            # loss, loss_atom, loss_coord, loss_dist
            pbar.set_postfix({k: f"{v:.4f}" for k, v in meter.mean().items()})
        sched.step()

        # -------- 验证 --------
        model.eval()
        meter_v = Meter()
        with torch.no_grad():
            for batch in tqdm(
                valid_loader,
                ncols=100,
                desc=f"Epoch {ep}/{cfg['epochs']} [valid]",
                dynamic_ncols=True,
            ):
                atom_ids = batch["atom_ids"].to(device)
                coords = batch["coords"].to(device)
                pad_mask = batch["pad_mask"].to(device)

                _, loss_dict, _ = model(atom_ids, coords, pad_mask)
                meter_v.update(loss_dict)

        val_mean = meter_v.mean()
        print(
            "[VALID] "
            + " ".join([f"{k}={v:.4f}" for k, v in val_mean.items()])
        )

        # -------- 保存共享编码器（最优验证 loss 时）--------
        if val_mean["loss"] < best_val:
            best_val = val_mean["loss"]
            no_improve = 0
            enc_path = os.path.join(cfg["out"], "shared_encoder.pt")
            torch.save(model.shared_encoder_state(), enc_path)
            print(f"√ 已保存共享编码器权重到：{enc_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"验证集连续 {patience} 个 epoch 未提升，提前停止训练。")
                break

        # 保存每个 epoch 的模型
        ckpt_path = os.path.join(cfg["out"], f"checkpoint_ep{ep}.pt")
        torch.save(
            {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "best_val": best_val,
            },
            ckpt_path,
        )

        # 顺手更新 latest checkpoint
        torch.save(
            {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "best_val": best_val,
            },
            checkpoint_path,
        )

    print("训练完成。输出目录：", cfg["out"])


if __name__ == "__main__":
    main()
