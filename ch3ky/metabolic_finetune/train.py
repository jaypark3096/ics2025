# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from rdkit import RDLogger
from tqdm.auto import tqdm

from .paths import CSV_PATH, ENZ_DIR, OUT_DIR, ENC_PATH, DEC_PATH
from .constants import (
    PAD_ID, BATCH_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY,
    NOISE_STD, MAX_ATOMS, NUM_WORKERS, DEVICE, EVAL_SPLIT,
    D_LR, D_STEPS, LAMBDA_ADV, LAMBDA_TRI,
)
from .dataset import MetabolicDataset, collate_fn
from .metabolic_model import MetabolicFinetuneModel
from .discriminators import MultiTaskDiscriminator

RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')


def _load_ckpt_or_die(model: MetabolicFinetuneModel):
    """从化学反应预训练阶段加载共享编码器/解码器权重."""
    assert ENC_PATH.is_file(), f"共享编码器缺失: {ENC_PATH}"
    assert DEC_PATH.is_file(), f"共享解码器缺失: {DEC_PATH}"
    enc_state = torch.load(str(ENC_PATH), map_location="cpu")
    if isinstance(enc_state, dict) and "state_dict" in enc_state:
        enc_state = enc_state["state_dict"]
    dec_state = torch.load(str(DEC_PATH), map_location="cpu")
    if isinstance(dec_state, dict) and "state_dict" in dec_state:
        dec_state = dec_state["state_dict"]
    model.load_shared_encoder(enc_state)
    model.load_shared_decoder(dec_state)
    print(f"[Info] 加载共享编码器: {ENC_PATH}")
    print(f"[Info] 加载共享解码器: {DEC_PATH}")


def train_and_eval():
    print(f"[Data] 读取: {CSV_PATH}")
    print(f"[Enzyme] 目录: {ENZ_DIR}")
    print(f"[Shared] Encoder: {ENC_PATH}")
    print(f"[Shared] Decoder: {DEC_PATH}")
    print(f"[Config] device={DEVICE}, batch_size={BATCH_SIZE}, epochs={NUM_EPOCHS}")

    # -------- 数据集 & DataLoader --------
    ds_train = MetabolicDataset(CSV_PATH, split="train", eval_ratio=EVAL_SPLIT)
    ds_eval  = MetabolicDataset(CSV_PATH, split="eval",  eval_ratio=EVAL_SPLIT)

    dl_train = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=lambda b: collate_fn(b, PAD_ID, MAX_ATOMS),
        pin_memory=True,
    )
    dl_eval = DataLoader(
        ds_eval,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=lambda b: collate_fn(b, PAD_ID, MAX_ATOMS),
        pin_memory=True,
    )

    # -------- 模型 & 判别器 & 预训练权重 --------
    model = MetabolicFinetuneModel(noise_std=NOISE_STD).to(DEVICE)
    _load_ckpt_or_die(model)

    disc = MultiTaskDiscriminator(hidden=model.d_model).to(DEVICE)

    opt_G = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    opt_D = AdamW(disc.parameters(), lr=D_LR, weight_decay=WEIGHT_DECAY)

    bce_logits = F.binary_cross_entropy_with_logits

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- 训练循环 ----------------
    for ep in range(1, NUM_EPOCHS + 1):
        model.train()
        disc.train()
        running_sup = 0.0
        running_g_total = 0.0

        pbar = tqdm(dl_train, ncols=100, desc=f"[Train] epoch {ep}/{NUM_EPOCHS}")
        for it, batch in enumerate(pbar, 1):
            # 把 batch 搬到 GPU
            for k in [
                "atom_s", "xyz_s", "pad_s",
                "atom_p", "xyz_p", "pad_p",
                "enzyme", "enzyme_mask",
            ]:
                batch[k] = batch[k].to(DEVICE, non_blocking=True)

            B = batch["atom_s"].size(0)
            if B < 2:
                # 太小的 batch 对三元组负例不友好，跳过 GAN 部分，只做重建
                loss_sup, ld, _ = model(batch, need_repr_for_gan=False)
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

            # ------- 1) 更新判别器 D -------
            for _ in range(D_STEPS):
                with torch.no_grad():
                    _, _, aux = model(batch, need_repr_for_gan=True)
                sub_vec       = aux["sub_vec"]        # (B,D)
                prod_fake_vec = aux["prod_fake_vec"]  # (B,D)
                prod_real_vec = aux["prod_real_vec"]  # (B,D)
                enz_vec       = aux["enz_vec"]        # (B,De)

                # 构造负样本：简单用 roll 做打乱
                sub_neg  = sub_vec.roll(shifts=1, dims=0)
                enz_neg  = enz_vec.roll(shifts=1, dims=0)
                prod_neg = prod_real_vec.roll(shifts=1, dims=0)

                ones  = torch.ones(B, device=DEVICE)
                zeros = torch.zeros(B, device=DEVICE)

                # 三元组一致性判别器
                # D_t1: 替换底物
                logits_t1_pos = disc.tri1_logits(sub_vec, enz_vec, prod_real_vec)
                logits_t1_neg = disc.tri1_logits(sub_neg, enz_vec, prod_real_vec)

                # D_t2: 替换酶
                logits_t2_pos = disc.tri2_logits(sub_vec, enz_vec, prod_real_vec)
                logits_t2_neg = disc.tri2_logits(sub_vec, enz_neg, prod_real_vec)

                # D_t3: 替换代谢物
                logits_t3_pos = disc.tri3_logits(sub_vec, enz_vec, prod_real_vec)
                logits_t3_neg = disc.tri3_logits(sub_vec, enz_vec, prod_neg)

                loss_t1 = 0.5 * (bce_logits(logits_t1_pos, ones) + bce_logits(logits_t1_neg, zeros))
                loss_t2 = 0.5 * (bce_logits(logits_t2_pos, ones) + bce_logits(logits_t2_neg, zeros))
                loss_t3 = 0.5 * (bce_logits(logits_t3_pos, ones) + bce_logits(logits_t3_neg, zeros))
                loss_tri = (loss_t1 + loss_t2 + loss_t3) / 3.0

                # 对抗判别器：真实 vs 生成
                logits_adv_real = disc.adv_logits(prod_real_vec)
                logits_adv_fake = disc.adv_logits(prod_fake_vec.detach())
                loss_adv_D = 0.5 * (bce_logits(logits_adv_real, ones) + bce_logits(logits_adv_fake, zeros))

                loss_D = loss_tri + loss_adv_D

                opt_D.zero_grad(set_to_none=True)
                loss_D.backward()
                opt_D.step()

            # ------- 2) 更新生成器 G -------
            loss_sup, ld, aux = model(batch, need_repr_for_gan=True)
            sub_vec       = aux["sub_vec"]        # (B,D)
            prod_fake_vec = aux["prod_fake_vec"]  # (B,D)
            enz_vec       = aux["enz_vec"]        # (B,De)

            ones = torch.ones(B, device=DEVICE)

            # 生成器的对抗损失：希望 D_adv 把生成样本判为真实
            logits_adv_fake = disc.adv_logits(prod_fake_vec)
            loss_g_adv = bce_logits(logits_adv_fake, ones)

            # 生成器的三元组一致性损失：希望 (drug, enzyme, fake_met) 被 D_t3 判为一致
            logits_t3_fake = disc.tri3_logits(sub_vec, enz_vec, prod_fake_vec)
            loss_g_tri = bce_logits(logits_t3_fake, ones)

            loss_G = loss_sup + LAMBDA_ADV * loss_g_adv + LAMBDA_TRI * loss_g_tri

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

        # —— 每个 epoch 保存一次当前 encoder/decoder —— #
        torch.save(model.shared_encoder_state(), OUT_DIR / f"shared_encoder_meta_ep{ep}.pt")
        torch.save(model.shared_decoder_state(), OUT_DIR / f"shared_decoder_meta_ep{ep}.pt")
        print(f"[Save] 已保存共享编码器/解码器到 {OUT_DIR}  (tag=meta_ep{ep})")

        # ---------------- 验证 (只看重建 loss) ----------------
        model.eval()
        val_loss, val_cnt = 0.0, 0
        with torch.no_grad():
            pbar_eval = tqdm(dl_eval, ncols=100, desc=f"[Eval ] epoch {ep}", leave=False)
            for batch in pbar_eval:
                for k in [
                    "atom_s", "xyz_s", "pad_s",
                    "atom_p", "xyz_p", "pad_p",
                    "enzyme", "enzyme_mask",
                ]:
                    batch[k] = batch[k].to(DEVICE, non_blocking=True)
                loss, _, _ = model(batch, need_repr_for_gan=False)
                val_loss += float(loss.item())
                val_cnt += 1
        if val_cnt > 0:
            print(f"[Eval ep{ep}] val_loss={val_loss / val_cnt:.4f}")

    # 最终导出一份“最终微调后的共享编码器/解码器”
    torch.save(model.shared_encoder_state(), OUT_DIR / "shared_encoder_finetuned.pt")
    torch.save(model.shared_decoder_state(), OUT_DIR / "shared_decoder_finetuned.pt")
    print(
        "[Done] 微调完成。最终权重：\n"
        f"  {OUT_DIR / 'shared_encoder_finetuned.pt'}\n"
        f"  {OUT_DIR / 'shared_decoder_finetuned.pt'}"
    )
