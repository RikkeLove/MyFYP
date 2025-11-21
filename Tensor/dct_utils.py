#dct_utils.py

import torch
from scipy.fftpack import dct
import numpy as np

class DCTMatrix:
    def __init__(self, n):
        """
        初始化一个 n x n的 DCT 矩阵
        :param n: 矩阵大小
        """
        self.n = n
        # 使用 scipy 生成 DCT 矩阵 (正交归一化)
        self.matrix = torch.tensor(dct(np.eye(n), norm='ortho'), dtype=torch.float32)

    def forward(self, x, dim):
        """
        对张量 x 在第 dim 维上应用 DCT 变换。
        :param x: 输入张量。
        :param dim: 应用 DCT 的维度。
        :return: 输出张量。
        """

        # 将第 dim 维移动到最前面
        perm = list(range(x.dim()))
        perm.insert(0, perm.pop(dim - 1))
        x_permuted = x.permute(perm)

        # 重塑成矩阵
        row_size = self.n
        col_size = int(torch.prod(torch.tensor(list(x.shape)[:dim - 1] + list(x.shape)[dim:])))
        x_matrix = x_permuted.reshape(row_size, -1)

        # 执行 DCT 变换
        y_matrix = torch.matmul(self.matrix.to(x.device), x_matrix.to(x.device))

        # 重塑回张量
        new_sz = [self.n] + list(x.shape)[:dim - 1] + list(x.shape)[dim:]
        y_permuted = y_matrix.reshape(new_sz)

        # 恢复维度顺序
        inv_perm = [0] * x.dim()
        for i, p in enumerate(perm):
            inv_perm[p] = i
        y = y_permuted.permute(inv_perm)

        return y

    def inverse(self, x, dim):
        """
        对张量 x 在第 dim 维上应用逆 DCT 变换。
        :param x: 输入张量。
        :param dim: 应用逆 DCT 的维度。
        :return: 输出张量。
        """
        # 逆 DCT 矩阵是 DCT 矩阵的转置
        inv_matrix = self.matrix.t()

        # 将第 dim 维移动到最前面
        perm = list(range(x.dim()))
        perm.insert(0, perm.pop(dim - 1))
        x_permuted = x.permute(perm)

        # 重塑成矩阵
        row_size = self.n
        col_size = int(torch.prod(torch.tensor(list(x.shape)[:dim - 1] + list(x.shape)[dim:])))
        x_matrix = x_permuted.reshape(row_size, -1)

        # 执行逆 DCT 变换
        y_matrix = torch.matmul(inv_matrix.to(x.device), x_matrix.to(x.device))

        # 重塑回张量
        new_sz = [self.n] + list(x.shape)[:dim - 1] + list(x.shape)[dim:]
        y_permuted = y_matrix.reshape(new_sz)

        # 恢复维度顺序
        inv_perm = [0] * x.dim()
        for i, p in enumerate(perm):
            inv_perm[p] = i
        y = y_permuted.permute(inv_perm)

        return y