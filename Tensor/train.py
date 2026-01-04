import torch
from torch.utils.data import DataLoader
from dataset import dataset
from MyNet import GTSNet
from GTNetLoss import GTNetLoss  # 用 GTNetLoss 作为损失函数
import argparse
import scipy.io as sio


def train(opt):
    # 数据集加载
    CAVE = sio.loadmat(opt.CAVE_path)['CAVE']
    KAIST = sio.loadmat(opt.KAIST_path)['KAIST']
    train_set = dataset(opt, CAVE, KAIST)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)

    # 模型
    model = GTSNet(input_shape=(opt.size, opt.size, 28), T=1).to(opt.device)
    loss_fn = GTNetLoss(alpha=0.005, beta=10.0, gamma=0.9).to(opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # 训练过程
    model.train()
    for epoch in range(opt.num_epochs):
        running_loss = 0.0
        for i, (input, label, mask_3d, mask_3d_shift, mask_3d_shift_s) in enumerate(train_loader):
            input, label = input.to(opt.device), label.to(opt.device)

            # 前向传播
            optimizer.zero_grad()
            _, se, sb = model(s=input)

            # 计算损失
            loss = loss_fn(se, sb, label)

            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{opt.num_epochs}], Loss: {running_loss/len(train_loader)}")

    print("Finished Training")


def test(opt):
    # 数据集加载
    CAVE = sio.loadmat(opt.CAVE_path)['CAVE']
    KAIST = sio.loadmat(opt.KAIST_path)['KAIST']
    test_set = dataset(opt, CAVE, KAIST)
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False)

    # 模型
    model = GTSNet(input_shape=(opt.size, opt.size, 28), T=1).to(opt.device)
    model.load_state_dict(torch.load(opt.model_path))  # 加载训练好的模型

    # 测试过程
    model.eval()
    with torch.no_grad():
        for i, (input, label, mask_3d, mask_3d_shift, mask_3d_shift_s) in enumerate(test_loader):
            input, label = input.to(opt.device), label.to(opt.device)

            # 前向传播
            _, se, sb = model(s=input)

            # 计算损失
            loss = GTNetLoss(alpha=0.005, beta=10.0, gamma=0.9).to(opt.device)(se, sb, label)
            print(f"Test Loss for batch {i+1}: {loss.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test GTSNet")
    parser.add_argument("--CAVE_path", type=str, required=True, help="Path to CAVE dataset")
    parser.add_argument("--KAIST_path", type=str, required=True, help="Path to KAIST dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--size", type=int, default=64, help="Image size (default: 64)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs (default: 50)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for training/testing (default: cuda)")

    opt = parser.parse_args()

    # 训练或测试
    if opt.isTrain:
        train(opt)
    else:
        test(opt)
