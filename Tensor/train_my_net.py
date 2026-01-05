import os
import random
import argparse
import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

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


def main():
    # Tensor 目录
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录（MyFYP）
    PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
    DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

    default_cave = os.path.join(DATASET_DIR, "CAVE_512_28")
    default_kaist = os.path.join(DATASET_DIR, "KAIST_CVPR2021")
    default_save = os.path.join(THIS_DIR, "checkpoints_gtsnet")

    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--cave_dir", type=str, default=default_cave)
    ap.add_argument("--kaist_dir", type=str, default=default_kaist)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--steps_per_epoch", type=int, default=2000)
    ap.add_argument("--use_kaist_prob", type=float, default=0.5)
    ap.add_argument("--no_aug", action="store_true")

    # train
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # model
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--use_dct", action="store_true")
    ap.add_argument("--D", type=int, default=4)
    ap.add_argument("--C", type=int, default=3)
    ap.add_argument("--G0", type=int, default=30)
    ap.add_argument("--G", type=int, default=12)

    # loss
    ap.add_argument("--alpha", type=float, default=0.005)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--gamma", type=float, default=0.9)

    # save
    ap.add_argument("--save_dir", type=str, default=default_save)
    ap.add_argument("--save_every", type=int, default=10)

    args = ap.parse_args()

    print("[Paths]")
    print("  PROJECT_DIR:", PROJECT_DIR)
    print("  cave_dir   :", os.path.abspath(args.cave_dir))
    print("  kaist_dir  :", os.path.abspath(args.kaist_dir))
    print("  save_dir   :", os.path.abspath(args.save_dir))

    seed_torch(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # Dataset -> returns [H,W,C], loader -> [B,H,W,C]
    ds = HSICubeDatasetBHWC(
        cave_dir=args.cave_dir,
        kaist_dir=args.kaist_dir,
        patch_size=args.patch,
        length=args.steps_per_epoch,
        use_kaist_prob=args.use_kaist_prob,
        enable_aug=(not args.no_aug),
        cache_size=2,
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(args.num_workers > 0),
    )

    # Model expects BHWC input; training uses model(s=gt) :contentReference[oaicite:2]{index=2}
    model = GTSNet(
        input_shape=(args.patch, args.patch, 28),
        output_shape=None,
        T=args.T,
        use_dct=args.use_dct,
        D=args.D, C=args.C, G0=args.G0, G=args.G
    ).to(device)

    # Loss supports dataformat="BHWC" :contentReference[oaicite:3]{index=3}
    loss_fn = GTNetLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma, dataformat="BHWC").to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250], gamma=0.5)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for it, gt in enumerate(loader, start=1):
            gt = gt.to(device, non_blocking=True)  # [B,H,W,C]

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                y, se, sb = model(s=gt)          # training path :contentReference[oaicite:4]{index=4}
                loss = loss_fn(se, sb, gt)        # GTNetLoss forward :contentReference[oaicite:5]{index=5}

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach())

            if it % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"[E{epoch:03d}] it {it:04d}/{len(loader)}  loss={epoch_loss/it:.6f}  lr={lr:.2e}")

        scheduler.step()

        if (epoch % args.save_every == 0) or (epoch == args.epochs):
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_path = os.path.join(args.save_dir, f"gtsnet_e{epoch:03d}_{stamp}.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
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
