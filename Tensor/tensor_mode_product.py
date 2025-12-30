# tensor_mode_product.py
import torch

def tensor_mode_product(X, A, n):
    """
        对张量 X 在第 n 维上与矩阵 A 进行模乘积运算。
        :param X: 输入张量，形状为 [d1, d2, ..., dN]
        :param A: 输入矩阵，形状为 [m, dn]
        :param n: 模乘积的维度，从 1 开始计数
        :return: 输出张量，形状为 [d1, ..., dn-1, m, dn+1, ..., dN]
    """

    #1.获取张量维度
    dims = X.dim()
    sz = list(X.shape)

    perm = list(range(dims))
    #2.将第n维度数据（索引n-1）移动到位置0
    perm.insert(0, perm.pop(n-1))
    X_permuted = X.permute(perm)

    #3.重塑矩阵
    row_size = sz[n-1]
    col_size = int(torch.prod(torch.tensor(sz[:n-1]+sz[n:])))
    X_matrix = X_permuted.reshape(row_size, col_size)

    #4.矩阵乘法
    Y_matrix = torch.matmul(A, X_matrix)

    #5.重塑回张量
    new_sz = [A.size(0)] + sz[:n-1] + sz[n:] #新的尺寸
    Y_permuted = Y_matrix.reshape(new_sz)

    #6.恢复维度顺序
    inv_perm = [0] * dims
    for i, p in enumerate(perm):
        inv_perm[p] = i
    Y = Y_permuted.permute(inv_perm)

    return Y

# ========================
# 测试用例
# ========================

if __name__ == "__main__":
    print("开始测试 tensor_mode_product 函数...")

    # Test Case 1: 3D Tensor
    X = torch.randn(2, 3, 4)
    A = torch.randn(5, 2)
    Y = tensor_mode_product(X, A, 1)
    assert Y.shape == (5, 3, 4), f"Test Case 1 Failed! Expected shape (5, 3, 4), got {Y.shape}"
    print("✅ Test Case 1: 3D Tensor passed.")

    # Test Case 2: 2D Tensor (Matrix)
    X = torch.randn(2, 3)
    A = torch.randn(4, 2)
    Y = tensor_mode_product(X, A, 1)
    assert Y.shape == (4, 3), f"Test Case 2 Failed! Expected shape (4, 3), got {Y.shape}"
    print("✅ Test Case 2: 2D Tensor passed.")

    # Test Case 3: 4D Tensor
    X = torch.randn(2, 3, 4, 5)
    A = torch.randn(6, 3)
    Y = tensor_mode_product(X, A, 2)
    assert Y.shape == (2, 6, 4, 5), f"Test Case 3 Failed! Expected shape (2, 6, 4, 5), got {Y.shape}"
    print("✅ Test Case 3: 4D Tensor passed.")

    print("\n🎉 所有测试用例均已通过！")