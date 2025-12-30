# sampling_layer.py

import torch
import torch.nn as nn
from tensor_mode_product import tensor_mode_product
from Tensor.dct_utils import DCTMatrix

class TensorSamplingLayer(nn.Module):

    def __init__(self, input_shape, output_shape=None, T=1, use_dct=False):
        """
        TensorSamplingLayer 实现 GTSNET 中的张量采样算子：
        S ×1 Φ1 ×2 Φ2 ×3 Φ3 并对 T 个可分离项求和
        """
        super(TensorSamplingLayer, self).__init__()

        self.input_shape = input_shape   # (H, W, C)
        self.output_shape = output_shape
        self.T = T
        self.use_dct = use_dct

        H, W, C = input_shape

        # 如果未指定输出尺寸，默认压缩为 0.5 倍
        if output_shape is None:
            output_shape = (H // 2, W // 2, C)
        self.output_shape = output_shape

        m1, m2, m3 = output_shape

        # ----------------------------
        #  ①：为每个 t 创建独立的 Φ矩阵
        # ----------------------------
        self.Phi1_list = nn.ParameterList([nn.Parameter(torch.randn(m1, H)) for _ in range(T)])
        self.Phi2_list = nn.ParameterList([nn.Parameter(torch.randn(m2, W)) for _ in range(T)])
        self.Phi3_list = nn.ParameterList([nn.Parameter(torch.randn(m3, C)) for _ in range(T)])

        # ----------------------------
        #  ②：创建 DCT（仅根据 H、W、C）
        # ----------------------------
        if use_dct:
            self.dct1 = DCTMatrix(H)
            self.dct2 = DCTMatrix(W)
            self.dct3 = DCTMatrix(C)

    def forward(self, x):
        """
        x: [batch, H, W, C]
        返回: [batch, m1, m2, m3]
        """

        B, H, W, C = x.shape

        # ----------------------------
        #  ③：只在循环外执行一次 DCT
        # ----------------------------
        if self.use_dct:
            x_dct = self.dct1.forward(x, 2)  # dim=2 → H 维
            x_dct = self.dct2.forward(x_dct, 3)  # dim=3 → W 维
            x_dct = self.dct3.forward(x_dct, 4)  # dim=4 → C 维
        else:
            x_dct = x

        # 初始化输出
        y = torch.zeros(B, *self.output_shape, device=x.device, dtype=x.dtype)

        # ----------------------------
        #  ④：对 T 个分支分别做 ×1, ×2, ×3
        # ----------------------------
        for t in range(self.T):
            Phi1 = self.Phi1_list[t]
            Phi2 = self.Phi2_list[t]
            Phi3 = self.Phi3_list[t]

            # 注意：dim=2,3,4 分别对应 H, W, C
            y_t = torch.einsum("bhwc,kh->bkwc", x_dct, Phi1)   # [B, m1, W, C]
            y_t = torch.einsum("bkwc,lw->bklc", y_t, Phi2)     # [B, m1, m2, C]
            y_t = torch.einsum("bklc,pc->bklp", y_t, Phi3)     # [B, m1, m2, m3]


            y += y_t

        return y
