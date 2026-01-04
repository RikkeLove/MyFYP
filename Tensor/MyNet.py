# gtsnet_model.py
import torch
import torch.nn as nn

from sampling_layer import TensorSamplingLayer
from coarse_reconstruction_layer import CoarseReconstructionLayer
from refinement_rdn import ResidualDenseBlock  # 你现有的RDB（输入输出: [B,30,H,W]）


class RDNRefinement(nn.Module):
    """
    Paper-style refinement module:
      - Shallow feature: Conv3x3(in->G0) + Conv3x3(G0->G0)
      - D x RDB (each keeps channel G0)
      - Global Feature Fusion: concat(D*G0) -> 1x1 -> 3x3 -> G0
      - Recon: Conv3x3(G0->in_channels)
      - Output: Sb = S_tilde + residual
    """
    def __init__(self, in_channels=3, D=4, C=3, G0=30, G=12):
        super().__init__()
        self.in_channels = in_channels
        self.D, self.C, self.G0, self.G = D, C, G0, G

        self.sfe1 = nn.Conv2d(in_channels, G0, kernel_size=3, padding=1, bias=True)
        self.sfe2 = nn.Conv2d(G0, G0, kernel_size=3, padding=1, bias=True)

        # 用你已有的 ResidualDenseBlock 来堆D次
        self.rdbs = nn.ModuleList([
            ResidualDenseBlock(InChannels=G0, OutChannels=G, KernelSize=3, LayerNumber=C)
            for _ in range(D)
        ])

        self.gff_1x1 = nn.Conv2d(D * G0, G0, kernel_size=1, padding=0, bias=True)
        self.gff_3x3 = nn.Conv2d(G0, G0, kernel_size=3, padding=1, bias=True)

        self.recon = nn.Conv2d(G0, in_channels, kernel_size=3, padding=1, bias=True)

    @staticmethod
    def bhwc_to_nchw(x):
        return x.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def nchw_to_bhwc(x):
        return x.permute(0, 2, 3, 1).contiguous()

    def forward(self, s_tilde_bhwc):
        x = self.bhwc_to_nchw(s_tilde_bhwc)  # [B,C,H,W]

        f1 = self.sfe1(x)   # [B,G0,H,W]
        f0 = self.sfe2(f1)  # [B,G0,H,W]

        rdb_outs = []
        h = f0
        for rdb in self.rdbs:
            h = rdb(h)          # [B,G0,H,W]
            rdb_outs.append(h)

        gff = self.gff_1x1(torch.cat(rdb_outs, dim=1))  # [B,G0,H,W]
        gff = self.gff_3x3(gff)                         # [B,G0,H,W]

        features = gff + f1
        residual = self.recon(features)                 # [B,C,H,W]
        out = x + residual                              # global residual learning

        return self.nchw_to_bhwc(out)


class GTSNet(nn.Module):
    """
    End-to-end:
      S (BHWC) -> Sampling -> Y
      Y -> Adjoint(Coarse) -> Se (proxy)
      Se -> Refinement(RDN) -> Sb (final)
    """
    def __init__(self, input_shape, output_shape=None, T=1, use_dct=False,
                 D=4, C=3, G0=30, G=12):
        super().__init__()
        self.sampling = TensorSamplingLayer(input_shape=input_shape, output_shape=output_shape, T=T, use_dct=use_dct)
        self.coarse = CoarseReconstructionLayer(self.sampling, use_dct=use_dct)
        self.refine = RDNRefinement(in_channels=input_shape[2], D=D, C=C, G0=G0, G=G)

    def forward(self, s=None, y=None):
        """
        用法：
          - 训练：forward(s=gt)  -> (y, se, sb)
          - 推理：forward(y=meas)-> (y, se, sb)
        """
        if y is None:
            if s is None:
                raise ValueError("Either s or y must be provided.")
            y = self.sampling(s)

        se = self.coarse(y)
        sb = self.refine(se)
        return y, se, sb

if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 1) 构建网络（注意：use_dct=False 可以先避开 DCT 导入路径问题）
    H, W, C = 64, 64, 3
    model = GTSNet(
        input_shape=(H, W, C),
        output_shape=None,   # 让 sampling_layer 用默认: (H//2, W//2, C)
        T=2,
        use_dct=True,       # 先用 False 跑通；你确认 DCT 导入没问题再改 True
        D=4, C=3, G0=30, G=12
    ).to(device)
    model.train()

    # 2) 随机输入（BHWC）
    S = torch.randn(2, H, W, C, device=device, dtype=torch.float32)

    # 3) 训练路径：给 s
    Y, Se, Sb = model(s=S)
    print("S  :", tuple(S.shape))
    print("Y  :", tuple(Y.shape))
    print("Se :", tuple(Se.shape))
    print("Sb :", tuple(Sb.shape))

    # 4) 推理路径：给 y（验证 forward(y=...) 也 OK）
    Y2, Se2, Sb2 = model(y=Y.detach())
    diff_se = (Se2 - Se.detach()).abs().mean().item()
    diff_sb = (Sb2 - Sb.detach()).abs().mean().item()
    print(f"Mean|Se(y)-Se(s)| = {diff_se:.6f}")
    print(f"Mean|Sb(y)-Sb(s)| = {diff_sb:.6f}")

    # 5) 反传测试：检查梯度能回到 sampling 和 refine
    loss = (Se - S).abs().mean() + (Sb - S).abs().mean()
    loss.backward()
    print("loss =", float(loss.detach()))

    # sampling 参数梯度
    print("Phi1 grad None? ", model.sampling.Phi1_list[0].grad is None)
    print("Phi2 grad None? ", model.sampling.Phi2_list[0].grad is None)
    print("Phi3 grad None? ", model.sampling.Phi3_list[0].grad is None)

    # refine 参数梯度
    print("recon.weight grad None? ", model.refine.recon.weight.grad is None)

    print("✅ GTSNet end-to-end forward/backward passed.")
