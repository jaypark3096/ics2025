# -*- coding: utf-8 -*-
import torch

# —— 模型与训练超参（与预训练保持一致）——
PAD_ID     = 0
VOCAB_SIZE = 120
D_MODEL    = 256
N_HEADS    = 8
D_FF       = 1024
LAYER_ENC  = 6
LAYER_DEC  = 4
RBF_K      = 64
RBF_MU_MAX = 10.0

# ---- 训练配置（生成器） ----
BATCH_SIZE   = 32
NUM_EPOCHS   = 6          # 训得更久一些
LR           = 3e-4        # 细调用的小学习率
WEIGHT_DECAY = 0.01

# 先把训练时的噪声关掉，让重建先学好
NOISE_STD    = 0.0
MAX_ATOMS    = 128
NUM_WORKERS  = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- GAN 相关 ----
D_LR        = 3e-4         # 判别器学习率
D_STEPS     = 1
LAMBDA_ADV  = 0.1          # 对抗损失权重（如需完全关掉可改成 0.0）
LAMBDA_TRI  = 0.1          # 三元组一致性损失权重（同上）

# ---- 评估设置 ----
TOPK        = 5
TOPK_LIST   = (5, 10, 12, 15)

EVAL_SPLIT  = 0.1
SEED        = 42

# 如果每个酶向量维度固定，也可手动填写；否则 dataset 会自动探测
ENZYME_DIM  = None

# ---- 三类监督损失的权重：先平一点 ----
LOSS_W_COORD = 1.0         # 坐标
LOSS_W_ATOM  = 1.0         # 原子类型
LOSS_W_BOND  = 1.0         # 成对距离 / bond bucket

# ---- 3D 命中判定相关 ----
# 论文中常见的 RMSD 命中阈值（单位 Å）：1.0 或 1.25
RMSD_THRESHOLDS      = (1.0, 1.25)
RMSD_DEFAULT_THRESH  = 1.25
