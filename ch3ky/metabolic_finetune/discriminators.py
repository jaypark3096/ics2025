# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn


class TripleConsistencyDiscriminator(nn.Module):
    """
    三元组一致性判别器 D_ti

    输入:
      sub_vec : (B, Ds)  底物表征
      enz_vec : (B, De)  酶表征
      met_vec : (B, Dm)  代谢物表征

    输出:
      logits  : (B,)     实数, 通过 sigmoid 后为“一致性概率”
    """
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self,
                sub_vec: torch.Tensor,
                enz_vec: torch.Tensor,
                met_vec: torch.Tensor) -> torch.Tensor:
        x = torch.cat([sub_vec, enz_vec, met_vec], dim=-1)  # (B, Ds+De+Dm)
        logits = self.net(x).squeeze(-1)                   # (B,)
        return logits


class AdversarialDiscriminator(nn.Module):
    """
    对抗判别器 D_adv
    只看代谢物特征, 判别其来自真实分布还是生成器
    """
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, met_vec: torch.Tensor) -> torch.Tensor:
        logits = self.net(met_vec).squeeze(-1)  # (B,)
        return logits


class MultiTaskDiscriminator(nn.Module):
    """
    多任务判别器:
      - tri1, tri2, tri3: 三个三元组一致性判别器
      - adv: 对抗判别器

    说明：
      - 结构保持不变，GAN 框架不改；
      - 训练脚本可以对同一个 (sub, enz, met) 构造多种正/负三元组，
        分别送入 tri1/tri2/tri3。
    """
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.tri1 = TripleConsistencyDiscriminator(hidden=hidden)
        self.tri2 = TripleConsistencyDiscriminator(hidden=hidden)
        self.tri3 = TripleConsistencyDiscriminator(hidden=hidden)
        self.adv  = AdversarialDiscriminator(hidden=hidden)

    def tri1_logits(self, sub_vec, enz_vec, met_vec):
        return self.tri1(sub_vec, enz_vec, met_vec)

    def tri2_logits(self, sub_vec, enz_vec, met_vec):
        return self.tri2(sub_vec, enz_vec, met_vec)

    def tri3_logits(self, sub_vec, enz_vec, met_vec):
        return self.tri3(sub_vec, enz_vec, met_vec)

    def adv_logits(self, met_vec):
        return self.adv(met_vec)
