# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn


class CGFF(nn.Module):
    """
    Conditionally Gated Feature Fusion（序列版）

    输入:
      drug_vec : (B, Dm)        —— 底物分子的全局向量
      aa_seq   : (B, L, De)     —— 酶的序列特征矩阵（预训练蛋白模型输出）
      aa_mask  : (B, L) bool    —— True 有效, False 为 pad

    输出:
      fused_vec: (B, Dm)        —— 融合后的向量，回到 d_model 维度

    说明：
      - 使用 gate 机制选择性关注酶序列中与当前底物相关的位置；
      - 最终输出仍是 d_model 维度的向量，可直接与分子 encoder 输出拼接/相加；
      - 保持接口不变，方便现有训练脚本使用。
    """
    def __init__(self, d_model: int, gate_dim: int = 256, out_dim: int | None = None):
        super().__init__()
        if out_dim is None:
            out_dim = d_model

        self.proj_drug = nn.Linear(d_model, gate_dim)
        self.proj_aa   = nn.LazyLinear(gate_dim)  # 首次 forward 自动推断 De
        self.gate_fc   = nn.LazyLinear(1)         # 每个 token 的 gate 标量

        self.fuse = nn.Sequential(
            nn.Linear(d_model + gate_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim)
        )

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        """
        x:(B,L,D), mask:(B,L) True=有效 → masked mean。
        """
        mask_f = mask.float().unsqueeze(-1)  # (B,L,1)
        s = (x * mask_f).sum(dim=dim)
        n = mask_f.sum(dim=dim).clamp_min(1e-6)
        return s / n

    def forward(self, drug_vec: torch.Tensor,
                aa_seq: torch.Tensor, aa_mask: torch.Tensor) -> torch.Tensor:
        # 1) 到 gate 空间
        d = self.proj_drug(drug_vec)                      # (B,G)
        a = self.proj_aa(aa_seq)                          # (B,L,G)

        # 2) 计算每个 token 的 gate 概率 p_j
        h = torch.tanh(d.unsqueeze(1) + a)                # (B,L,G)
        p = torch.sigmoid(self.gate_fc(h)).squeeze(-1)    # (B,L)

        # 3) soft gating（可微）：A' = A ⊙ p
        p = p * aa_mask.float()                            # 把 pad 置0
        a_gated = a * p.unsqueeze(-1)                      # (B,L,G)

        # 4) token 级汇聚（masked mean）
        a_mean = self.masked_mean(a_gated, aa_mask, dim=1) # (B,G)

        # 5) 融合到底物向量上
        fused = self.fuse(torch.cat([drug_vec, a_mean], dim=-1))  # (B,Dm)
        return fused
