# -*- coding: utf-8 -*-
# metabolic_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设这些组件在你的项目中已存在，保持引用路径一致
# 如果路径不同，请修改这里
from chemical_pretrain.model_components import (
    pairwise_distance,
    RBFEncoder,
    MLP
)
# 复用 Stage 1 的模型类，作为 Base
from chemical_pretrain.pretrain_model import MoleculePretrainModel
from dataset import PAD_ID  # 确保 dataset.py 里定义了 PAD_ID=0

try:
    from cgff import CGFF  # 你的条件门控融合模块
except ImportError:
    # 简单的 fallback，以防文件缺失
    class CGFF(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.fusion = nn.Linear(d_model * 2, d_model)

        def forward(self, sub, enz):
            return self.fusion(torch.cat([sub, enz], dim=-1))


class MetabolicFinetuneModel(nn.Module):
    def __init__(
            self,
            config_stage1,  # Stage 1 的配置参数 (dict)
            weights_path=None,  # shared_encoder.pt 的路径
            use_discriminator=True
    ):
        super().__init__()

        # 1. 加载 Stage 1 底座 (Shared Encoder / Decoder)
        # 我们直接实例化 Stage 1 模型，方便复用其 helper functions (如 rbf, pe 等)
        self.base = MoleculePretrainModel(**config_stage1)

        # 加载预训练权重
        if weights_path:
            print(f"[MetabolicModel] Loading pretrained weights from {weights_path}")
            ckpt = torch.load(weights_path, map_location='cpu')
            state_dict = ckpt.get('model_state_dict', ckpt)
            # 允许部分加载（比如 Stage 1 是 Encoder-Only 或 Decoder 结构不同）
            self.base.load_state_dict(state_dict, strict=False)

        d_model = config_stage1['d_model']
        vocab_size = config_stage1['vocab_size']

        # 2. 酶特征处理 (Assuming PLM features are input as (B, L_enz, D_plm))
        # 假设 PLM 输出维度是 1280 (ESM-1b) 或其他，需映射到 d_model
        self.plm_dim = 1280
        self.enz_proj = nn.Linear(self.plm_dim, d_model)

        # 3. CGFF 融合模块
        self.cgff = CGFF(d_model)  # 需要确保 cgff.py 里的定义匹配

        # 4. 噪声注入与桥接
        self.noise_proj = nn.Linear(d_model, d_model)

        # 5. 生成器 Query (Product Query)
        # 产物最大长度，需与 Dataset 中一致
        self.max_prod_len = config_stage1.get('max_len', 128)
        self.product_query = nn.Parameter(torch.randn(1, self.max_prod_len, d_model))

        # 6. 预测头 (Heads)
        # Atom Head 复用 self.base.head_atom
        # Dist Head 复用 self.base.dist_head
        # Coord Head: 必须是绝对坐标预测
        self.head_coord = nn.Linear(d_model, 3)

        # 7. 判别器相关 (GAN)
        self.use_discriminator = use_discriminator

        # 8. 损失权重 (可调整)
        self.w_atom = 1.0
        self.w_coord = 2.0  # 坐标通常数值小，权重给大点
        self.w_dist = 1.0

    @staticmethod
    def _aligned_coord_mse(pred_coord, true_coord, pad_mask):
        """
        核心修正：Kabsch 算法对齐后计算 MSE Loss
        pred_coord: (B, N, 3)
        true_coord: (B, N, 3)
        pad_mask: (B, N) True=Pad
        """
        loss_total = torch.tensor(0.0, device=pred_coord.device)
        batch_size = pred_coord.size(0)
        valid_count = 0

        for i in range(batch_size):
            # 获取非 Pad 的有效原子掩码
            valid = ~pad_mask[i]  # (N,)
            if valid.sum() < 3:
                # 原子太少无法做 SVD，直接算 MSE (或跳过)
                diff = pred_coord[i][valid] - true_coord[i][valid]
                loss_total += (diff ** 2).mean()
                valid_count += 1
                continue

            P = pred_coord[i][valid].double()  # 预测 (M, 3)
            Q = true_coord[i][valid].double()  # 真实 (M, 3)

            # 1. 去中心化
            P_mean = P.mean(dim=0, keepdim=True)
            Q_mean = Q.mean(dim=0, keepdim=True)
            P_c = P - P_mean
            Q_c = Q - Q_mean

            # 2. 协方差矩阵 H
            H = P_c.T @ Q_c  # (3, 3)

            # 3. SVD
            try:
                U, S, Vh = torch.linalg.svd(H)
            except RuntimeError:
                # SVD 失败 fallback
                loss_total += F.mse_loss(P.float(), Q.float())
                valid_count += 1
                continue

            # 4. 旋转矩阵 R
            R = Vh.T @ U.T

            # 5. 反射修正 (如果行列式为负)
            if torch.det(R) < 0:
                Vh[2, :] *= -1
                R = Vh.T @ U.T

            # 6. 旋转 P
            P_rot = P_c @ R

            # 7. 计算 Loss (P_rot 应该接近 Q_c)
            loss_total += F.mse_loss(P_rot.float(), Q_c.float())
            valid_count += 1

        return loss_total / max(1, valid_count)

    def forward(self, batch_data, need_gan_features=False):
        """
        batch_data: Dict from DataLoader
        包含:
          - sub_ids, sub_coords, sub_mask (底物)
          - enz_feat (酶 PLM 特征)
          - prod_ids, prod_coords, prod_mask (产物 GT)
        """
        # Unpack
        sub_ids = batch_data['sub_ids']  # (B, N_sub)
        sub_coords = batch_data['sub_coords']  # (B, N_sub, 3)
        sub_mask = batch_data['sub_mask']  # (B, N_sub)

        enz_feat = batch_data['enz_feat']  # (B, L_enz, 1280)

        prod_ids = batch_data.get('prod_ids', None)  # (B, N_prod)
        prod_coords = batch_data.get('prod_coords', None)  # (B, N_prod, 3)
        prod_mask = batch_data.get('prod_mask', None)  # (B, N_prod)

        B = sub_ids.size(0)
        device = sub_ids.device

        # ====================
        # 1. Encode Substrate (Shared Encoder)
        # ====================
        x_sub = self.base.atom_emb(sub_ids)
        dist_sub = self.base.pairwise_distance(sub_coords, sub_mask)
        # RBF & Pair Bias
        e_ij_sub = self.base.rbf(dist_sub)
        b_ij_sub = self.base.rbf_mlp(e_ij_sub)
        valid_sub = ~sub_mask
        pair_valid_sub = valid_sub.unsqueeze(1) & valid_sub.unsqueeze(2)
        b0_sub = b_ij_sub * pair_valid_sub.unsqueeze(-1).float()

        x_enc, _ = self.base.encoder(x_sub, b0_sub, sub_mask)  # (B, N_sub, D)

        # ====================
        # 2. Encode Enzyme & Fusion (CGFF)
        # ====================
        # 映射酶特征
        x_enz = self.enz_proj(enz_feat)  # (B, L_enz, D)

        # Pooling 得到全局向量用于条件控制
        # Substrate Global
        mask_float = valid_sub.float().unsqueeze(-1)
        sub_global = (x_enc * mask_float).sum(1) / (mask_float.sum(1) + 1e-6)  # (B, D)

        # Enzyme Global (Simple Mean)
        enz_global = x_enz.mean(dim=1)  # (B, D)

        # CGFF Fusion
        fused_context = self.cgff(sub_global, enz_global)  # (B, D)

        # Add Noise (GAN Generator Style)
        z = torch.randn_like(fused_context)
        latent = fused_context + self.noise_proj(z)  # (B, D)

        # ====================
        # 3. Decode Product
        # ====================
        # Query Expansion
        dec_in = self.product_query.expand(B, -1, -1)  # (B, Np, D)

        # Condition
        dec_in = dec_in + latent.unsqueeze(1)

        # PE
        dec_in = self.base.pe(dec_in)

        # Decoder
        b_dec_init = torch.zeros(B, self.max_prod_len, self.max_prod_len, self.base.n_heads, device=device)
        fake_mask = torch.zeros(B, self.max_prod_len, dtype=torch.bool, device=device)

        x_dec, b_dec = self.base.decoder(dec_in, b_dec_init, fake_mask)

        # ====================
        # 4. Prediction Heads
        # ====================
        logits_atom = self.base.head_atom(x_dec)  # (B, Np, V)
        pred_coords = self.head_coord(x_dec)  # (B, Np, 3)
        pred_dist = self.base.dist_head(b_dec)  # (B, Np, Np)

        # ====================
        # 5. Loss Calculation
        # ====================
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device)

        if prod_ids is not None:
            # 截断预测以匹配 Target 长度
            T = min(self.max_prod_len, prod_ids.size(1))

            # Slicing
            p_logits = logits_atom[:, :T, :]
            p_coords = pred_coords[:, :T, :]
            p_dist = pred_dist[:, :T, :T]

            t_ids = prod_ids[:, :T]
            t_coords = prod_coords[:, :T, :]
            t_mask = prod_mask[:, :T]  # True=Pad

            valid_p = ~t_mask

            # --- Loss 1: Atom (CE) ---
            l_atom = F.cross_entropy(
                p_logits.reshape(-1, p_logits.size(-1)),
                t_ids.reshape(-1),
                ignore_index=PAD_ID
            )

            # --- Loss 2: Coord (Kabsch Aligned MSE) ---
            # 这里的 t_mask 是 True for Pad，传入函数需注意
            l_coord = self._aligned_coord_mse(p_coords, t_coords, t_mask)

            # --- Loss 3: Dist (MSE) ---
            t_dist_mat = pairwise_distance(t_coords, t_mask)
            pair_mask = valid_p.unsqueeze(1) & valid_p.unsqueeze(2)
            l_dist = F.smooth_l1_loss(p_dist[pair_mask], t_dist_mat[pair_mask])

            # Weighted Sum
            total_loss = (self.w_atom * l_atom +
                          self.w_coord * l_coord +
                          self.w_dist * l_dist)

            loss_dict = {
                "loss": total_loss,
                "l_atom": l_atom,
                "l_coord": l_coord,
                "l_dist": l_dist
            }

        # 辅助输出 (给 GAN 或 可视化用)
        aux_out = {
            "logits_atom": logits_atom,
            "pred_coords": pred_coords,
            "latent": latent  # 用于 Discriminator 判别
        }

        return total_loss, loss_dict, aux_out