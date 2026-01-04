#GTNetLoss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GTNetLoss(nn.Module):
    """
       L(Se, Sb, S) = L_proxy(Se,S) + L_final(Sb,S)
       L_proxy: mean L1
       L_final: mean L1 + alpha * Rb(Sb,S)
       Rb: weighted sparse gradient prior (paper eq.18)
    """
    def __init__(self, alpha = 0.005, beta = 10.0, gamma = 0.9, dataformat = "BHWC", eps=1e-12):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.data_format = dataformat.upper()
        self.eps = eps

    def _to_nchw(self, x):
        if self.data_format == "BHWC":
            return x.permute(0, 3, 1, 2).contiguous()
        elif self.data_format == "NCHW":
            return x
        else:
            raise ValueError("data_format must be 'BHWC' or 'NCHW'")

    @staticmethod
    def _grad_h(x):# along height (n1)
        return x[:, :, 1:, :] - x[:, :, :-1, :]

    @staticmethod
    def _grad_w(x):  # along width (n2)
        # x: [B,C,H,W] -> [B,C,H,W-1]
        return x[:, :, :, 1:] - x[:, :, :, :-1]

    def _rb(self, sb, s_gt):
        """
        Rb(Sb,S) = sum exp(-beta*|∇S|^gamma)*|∇Sb|^gamma (both directions)
        We return batch-mean (so scale is stable).
        """

        #gradients
        gh_gt = self._grad_h(s_gt)
        gw_gt = self._grad_w(s_gt)
        gh_sb = self._grad_h(sb)
        gw_sb = self._grad_w(sb)

        # weights from GT (edge-aware)
        w_h = torch.exp(-self.beta * (gh_gt.abs() + self.eps).pow(self.gamma))
        w_w = torch.exp(-self.beta * (gw_gt.abs() + self.eps).pow(self.gamma))

        # penalty on prediction gradients
        p_h = (gh_sb.abs() + self.eps).pow(self.gamma)
        p_w = (gw_sb.abs() + self.eps).pow(self.gamma)

        rb = (w_h * p_h).mean() + (w_w * p_w).mean()
        return rb

    def forward(self, se, sb, s_gt):
        """
        se, sb, s_gt: BHWC or NCHW (controlled by data_format)
        returns: scalar loss
        """
        se = self._to_nchw(se)
        sb = self._to_nchw(sb)
        s_gt = self._to_nchw(s_gt)

        # proxy L1 (eq.16)
        l_proxy = F.l1_loss(se, s_gt, reduction="mean")

        # final L1 (eq.17 main term)
        l_final_l1 = F.l1_loss(sb, s_gt, reduction="mean")

        # regularizer (eq.18)
        reg = self._rb(sb, s_gt)

        # total (eq.17 + eq.16)
        l_final = l_final_l1 + self.alpha * reg
        return l_proxy + l_final


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    loss_fn = GTNetLoss(dataformat="BHWC").to(device)

    B, H, W, C = 2, 64, 64, 3
    se = torch.randn(B, H, W, C, device=device, requires_grad=True)
    sb = torch.randn(B, H, W, C, device=device, requires_grad=True)
    gt = torch.randn(B, H, W, C, device=device)

    loss = loss_fn(se, sb, gt)
    print("loss =", float(loss))

    loss.backward()
    print("se.grad None?", se.grad is None)
    print("sb.grad None?", sb.grad is None)
    print("✅ forward/backward passed.")
