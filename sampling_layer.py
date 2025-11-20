#sampling_layer.py

import torch
import torch.nn as nn
from .tensor_mode_product import tensor_mode_product
from .dct_utils import DCTMatrix

class TensorSamplingLayer(nn.Module):
    def __init__(self, input_shape, output_shape=None, T=1, use_dct=False):
        """
        初始化 TensorSamplingLayer。
        :param input_shape: 输入张量的形状，例如 (H, W, B)。
        :param output_shape: 输出张量的形状，例如 (m1, m2, m3)。
        :param T: 张量项的数量，默认为 1。
        :param use_dct: 是否使用 DCT 变换矩阵。
        """

        super(TensorSamplingLayer, self).__init__()
