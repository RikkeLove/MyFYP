import torch.utils.data as tud
import random
import torch
import numpy as np
import scipy.io as sio


class dataset(tud.Dataset):
    def __init__(self, isTrain, size, trainset_num, testset_num, CAVE_path, KAIST_path, mask_path):
        super(dataset, self).__init__()
        self.isTrain = isTrain
        self.size = size
        if self.isTrain:
            self.num = trainset_num
        else:
            self.num = testset_num

        # 直接在这里指定数据集路径并加载数据
        self.CAVE = sio.loadmat(CAVE_path)['dataset/CAVE_512_28']
        self.KAIST = sio.loadmat(KAIST_path)['KAIST']

        # 加载 mask 数据
        data = sio.loadmat(mask_path)
        self.mask = data['mask']
        self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 28))

    def __getitem__(self, index):
        if self.isTrain:
            # index1 = 0
            index1 = random.randint(0, 29)
            d = random.randint(0, 1)
            if d == 0:
                hsi = self.CAVE[:, :, :, index1]
            else:
                hsi = self.KAIST[:, :, :, index1]
        else:
            index1 = index
            hsi = self.CAVE[:, :, :, index1]  # 使用 CAVE 测试集

        shape = np.shape(hsi)

        # 随机裁剪数据
        px = random.randint(0, shape[0] - self.size)
        py = random.randint(0, shape[1] - self.size)
        label = hsi[px:px + self.size, py:py + self.size, :]

        # 随机裁剪 mask 数据
        pxm = random.randint(0, 660 - self.size)
        pym = random.randint(0, 660 - self.size)
        mask_3d = self.mask_3d[pxm:pxm + self.size, pym:pym + self.size, :]

        # 位移后的 mask
        mask_3d_shift = np.zeros((self.size, self.size + (28 - 1) * 2, 28))
        mask_3d_shift[:, 0:self.size, :] = mask_3d
        for t in range(28):
            mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2 * t, axis=1)
        mask_3d_shift_s = np.sum(mask_3d_shift ** 2, axis=2, keepdims=False)
        mask_3d_shift_s[mask_3d_shift_s == 0] = 1

        # 数据增强（只对 label 做）
        if self.isTrain:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)

            # 随机旋转
            for j in range(rotTimes):
                label = np.rot90(label)

            # 随机上下翻转
            for j in range(vFlip):
                label = label[:, ::-1, :].copy()

            # 随机左右翻转
            for j in range(hFlip):
                label = label[::-1, :, :].copy()

        # 计算 input：通过 mask 生成测量图
        temp = mask_3d * label
        temp_shift = np.zeros((self.size, self.size + (28 - 1) * 2, 28))
        temp_shift[:, 0:self.size, :] = temp
        for t in range(28):
            temp_shift[:, :, t] = np.roll(temp_shift[:, :, t], 2 * t, axis=1)
        meas = np.sum(temp_shift, axis=2)
        input = meas / 28 * 2 * 1.2

        # 使用二项分布生成噪声
        QE, bit = 0.4, 2048
        input = np.random.binomial((input * bit / QE).astype(int), QE)
        input = np.float32(input) / np.float32(bit)

        # 转换为 PyTorch tensor，并调整维度
        label = torch.FloatTensor(label.copy()).permute(2, 0, 1)
        input = torch.FloatTensor(input.copy())
        mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1)
        mask_3d_shift_s = torch.FloatTensor(mask_3d_shift_s.copy())

        return input, label, mask_3d, mask_3d_shift, mask_3d_shift_s

    def __len__(self):
        return self.num
