# coarse_reconstruction_layer.py

import torch
import torch.nn as nn
from dct_utils import DCTMatrix # 确保导入 DCTMatrix

class CoarseReconstructionLayer(nn.Module):
    """
    粗重建模块 (Coarse Estimation Module)。
    它是采样操作 (P) 的伴随 (adjoint)，即 P'。
    接收测量值 Y，输出初始估计 S~。
    S~ = sum_{t=1}^T (Y ×1 (Phi1_t.T) ×2 (Phi2_t.T) ×3 (Phi3_t.T))
    如果使用 DCT，则先在 DCT 域进行累加，最后再应用逆 DCT 转换回空间域。
    """

    def __init__(self, sampling_layer, use_dct=False):
        """
        :param sampling_layer: 一个 TensorSamplingLayer 实例，从中获取 Phi 矩阵参数。
        :param use_dct: 是否在重建前对测量值应用逆 DCT。
        """
        super(CoarseReconstructionLayer, self).__init__()

        self.sampling_layer = sampling_layer
        self.use_dct = use_dct

        # 获取原始输入形状 (H, W, C) 用于初始化逆 DCT
        H, W, C = sampling_layer.input_shape

        # 从采样层获取 T 和输出形状 (m1, m2, m3)
        self.T = sampling_layer.T
        m1, m2, m3 = sampling_layer.output_shape

        if use_dct:
            # 创建用于原信号尺寸的逆 DCT 矩阵
            # 这些是 *固定* 的变换矩阵，不参与学习
            self.dct1_inv = DCTMatrix(H)
            self.dct2_inv = DCTMatrix(W)
            self.dct3_inv = DCTMatrix(C)

    def forward(self, y):
        """
        :param y: 测量值张量，形状为 [batch, m1, m2, m3]
        :return: 粗重建结果 S~，形状为 [batch, H, W, C]
        """
        B, m1, m2, m3 = y.shape
        # 获取原始输入尺寸 (H, W, C) 用于初始化输出张量
        H, W, C = self.sampling_layer.input_shape

        # 初始化 DCT 域的输出 (如果 use_dct 为 True) 或 空间域输出 (如果 use_dct 为 False)
        s_tilde_dct_or_spatial = torch.zeros(B, H, W, C, device=y.device, dtype=y.dtype)

        # 对 T 个分支进行累加
        for t in range(self.sampling_layer.T):
            # 获取采样层中对应的 Phi 矩阵并转置
            Phi1_t = self.sampling_layer.Phi1_list[t]  # Shape: [m1, H]
            Phi2_t = self.sampling_layer.Phi2_list[t]  # Shape: [m2, W]
            Phi3_t = self.sampling_layer.Phi3_list[t]  # Shape: [m3, C]

            Phi1_t_transpose = Phi1_t.t()  # Shape: [H, m1]
            Phi2_t_transpose = Phi2_t.t()  # Shape: [W, m2]
            Phi3_t_transpose = Phi3_t.t()  # Shape: [C, m3]

            # 开始对当前分支 y 进行模乘积 (使用转置矩阵)
            # y: [B, m1, m2, m3]

            # Phi1^T: [H, m1]
            s_t_branch = torch.einsum("bklp,hk->bhlp", y, Phi1_t_transpose)  # -> [B, H, m2, m3]
            # Phi2^T: [W, m2]
            s_t_branch = torch.einsum("bhlp,wl->bhwp", s_t_branch, Phi2_t_transpose)  # -> [B, H, W, m3]
            # Phi3^T: [C, m3]
            s_t_branch = torch.einsum("bhwp,cp->bhwc", s_t_branch, Phi3_t_transpose)  # -> [B, H, W, C]

            # 将当前分支结果累加到总和中 (此时仍在 DCT 域或空间域)
            s_tilde_dct_or_spatial += s_t_branch

        # (可选) 如果使用 DCT，对累加后的 DCT 域结果应用逆 DCT，转换回空间域
        # 这是关键修改点：先累加，再逆变换
        if self.use_dct:
            # 注意：维度 dim=2, 3, 4 对应 H, W, C
            s_tilde_spatial = self.dct1_inv.inverse(s_tilde_dct_or_spatial, 2)  # Apply inverse DCT on H dimension
            s_tilde_spatial = self.dct2_inv.inverse(s_tilde_spatial, 3)  # Apply inverse DCT on W dimension
            s_tilde_spatial = self.dct3_inv.inverse(s_tilde_spatial, 4)  # Apply inverse DCT on C dimension
            return s_tilde_spatial
        else:
            # 如果不使用 DCT，则累加后的结果已经是空间域
            return s_tilde_dct_or_spatial
