# Tensor/option_my_net.py
import os
import argparse


def get_opt():
    # Tensor 目录
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录（MyFYP）
    PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
    DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

    default_cave = os.path.join(DATASET_DIR, "CAVE_512_28")
    default_kaist = os.path.join(DATASET_DIR, "KAIST_CVPR2021")
    default_save = os.path.join(THIS_DIR, "checkpoints_gtsnet")

    parser = argparse.ArgumentParser()

    # ---------------- data ----------------
    parser.add_argument("--cave_dir", type=str, default=default_cave)
    parser.add_argument("--kaist_dir", type=str, default=default_kaist)
    parser.add_argument("--patch", type=int, default=256)
    parser.add_argument("--steps_per_epoch", type=int, default=2000)
    parser.add_argument("--use_kaist_prob", type=float, default=0.5)
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--cache_size", type=int, default=2)

    # ---------------- train ----------------
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_every", type=int, default=50)

    # ---------------- model ----------------
    parser.add_argument("--T", type=int, default=2)
    parser.add_argument("--use_dct", action="store_true")
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--C", type=int, default=3)
    parser.add_argument("--G0", type=int, default=30)
    parser.add_argument("--G", type=int, default=12)

    # ---------------- loss ----------------
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.9)

    # ---------------- save ----------------
    parser.add_argument("--save_dir", type=str, default=default_save)
    parser.add_argument("--save_every", type=int, default=10)

    opt = parser.parse_args()

    # 统一转绝对路径，避免工作目录导致的路径错误
    opt.project_dir = PROJECT_DIR
    opt.cave_dir = os.path.abspath(opt.cave_dir)
    opt.kaist_dir = os.path.abspath(opt.kaist_dir)
    opt.save_dir = os.path.abspath(opt.save_dir)

    os.makedirs(opt.save_dir, exist_ok=True)

    return opt
