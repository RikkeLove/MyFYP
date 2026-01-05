# Tensor/train_my_net.py
import os
import random
import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from option_my_net import get_opt
from dataset import HSICubeDatasetBHWC, seed_worker
from MyNet import GTSNet
from GTNetLoss import GTNetLoss


def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def print_device_info(device: str):
    print("[Device]")
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"  Running on GPU: {name}")
        print(f"  CUDA version (torch): {torch.version.cuda}")
        print(f"  Total VRAM: {total_gb:.2f} GB")
    else:
        print("  Running on CPU")


def main():
    opt = get_opt()

    print("[Paths]")
    print("  PROJECT_DIR:", opt.project_dir)
    print("  cave_dir   :", opt.cave_dir)
    print("  kaist_dir  :", opt.kaist_dir)
    print("  save_dir   :", opt.save_dir)

    seed_torch(opt.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_device_info(device)

    # AMP 只在 GPU 上启用更合理
    amp_enabled = bool(opt.amp and device == "cuda")
    amp_device = "cuda" if device == "cuda" else "cpu"

    # Dataset -> returns [H,W,C], loader -> [B,H,W,C]
    ds = HSICubeDatasetBHWC(
        cave_dir=opt.cave_dir,
        kaist_dir=opt.kaist_dir,
        patch_size=opt.patch,
        length=opt.steps_per_epoch,
        use_kaist_prob=opt.use_kaist_prob,
        enable_aug=(not opt.no_aug),
        cache_size=opt.cache_size,
    )

    g = torch.Generator()
    g.manual_seed(opt.seed)

    loader = DataLoader(
        ds,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(opt.num_workers > 0),
    )

    model = GTSNet(
        input_shape=(opt.patch, opt.patch, 28),
        output_shape=None,
        T=opt.T,
        use_dct=opt.use_dct,
        D=opt.D, C=opt.C, G0=opt.G0, G=opt.G
    ).to(device)

    loss_fn = GTNetLoss(alpha=opt.alpha, beta=opt.beta, gamma=opt.gamma, dataformat="BHWC").to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 100, 150, 200, 250], gamma=0.5
    )

    # ✅ 新版 AMP：只在 GPU 上启用
    scaler = torch.amp.GradScaler(amp_device, enabled=amp_enabled)

    printed_once = False

    for epoch in range(1, opt.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for it, gt in enumerate(loader, start=1):
            gt = gt.to(device, non_blocking=True)  # [B,H,W,C]

            # 可选：只打印一次，确认模型/数据在哪个设备上
            if not printed_once:
                print("[Check]")
                print("  gt.device:", gt.device)
                print("  model.device:", next(model.parameters()).device)
                print("  AMP enabled:", amp_enabled)
                printed_once = True

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=amp_device, enabled=amp_enabled):
                y, se, sb = model(s=gt)
                loss = loss_fn(se, sb, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach())

            if it % opt.print_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"[E{epoch:03d}] it {it:04d}/{len(loader)}  loss={epoch_loss/it:.6f}  lr={lr:.2e}")

        scheduler.step()

        if (epoch % opt.save_every == 0) or (epoch == opt.epochs):
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_path = os.path.join(opt.save_dir, f"gtsnet_e{epoch:03d}_{stamp}.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "opt": vars(opt),
            }, ckpt_path)
            print("Saved:", ckpt_path)


if __name__ == "__main__":
    # Windows 多进程更稳
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    main()
