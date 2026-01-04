#refinement_rdn.py
import torch
import torch.nn as nn

class ResidualDenseBlock(nn.Module):
    """
    RDB块实现：
    对于F^{(d-1)}RDB块:
    张量形状:[B, 30, H, W]
    输入通道数：InputChannelG0 = 30
    卷积输出通道数：OutputChannelG = 12
    最终输出:F^{(d)} = F^{(d-1)} + LFF(...)
    """
    def __init__(self,InChannels = 30, OutChannels = 12, KernelSize = 3, LayerNumber = 3):
        super().__init__()

        self.InChannels = InChannels
        self.OutChannels = OutChannels
        self.KernelSize = KernelSize
        self.LayerNumber = LayerNumber
        self.LayerList = nn.ModuleList()

        for k in range(self.LayerNumber):
            InChannelskblock = self.InChannels + OutChannels * k # 每一个卷积层输入结果与
            self.LayerList.append(
                nn.Sequential(
                    nn.Conv2d(InChannelskblock, OutChannels, KernelSize, stride=1, padding=1),
                    nn.ReLU(inplace=True)
                )
            )

        self.lff = nn.Conv2d(InChannels + OutChannels * LayerNumber, InChannels,  kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        feats = [x]

        for i in range(self.LayerNumber):
            inp = torch.cat(feats, dim=1)
            out = self.LayerList[i](inp)
            feats.append(out)

        fused = self.lff(torch.cat(feats, dim=1)) # -> [B, 30, H, W]

        return x + fused

if __name__ == "__main__":
    import traceback
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 初始化
    rdb = ResidualDenseBlock(InChannels=30, OutChannels=12, KernelSize=3, LayerNumber=3).to(device)
    rdb.eval()

    # 随机输入：NCHW
    x = torch.randn(2, 30, 64, 64, device=device)
    print("Input shape:", tuple(x.shape))

    # 尝试前向 + 打印每层 shape
    try:
        with torch.no_grad():
            feats = [x]
            for i in range(rdb.LayerNumber):
                inp = torch.cat(feats, dim=1)
                out = rdb.LayerList[i](inp)
                print(f"[Layer {i+1}] inp={tuple(inp.shape)} -> out={tuple(out.shape)}")
                feats.append(out)

            cat_all = torch.cat(feats, dim=1)
            fused = rdb.lff(cat_all)
            y = x + fused

        print("Concat(all feats) shape:", tuple(cat_all.shape))
        print("Fused shape:", tuple(fused.shape))
        print("Output shape:", tuple(y.shape))
        print("✅ Forward ran successfully.")

    except Exception as e:
        print("❌ Forward failed.")
        print("Error:", repr(e))
        print("\nTraceback:")
        traceback.print_exc()

