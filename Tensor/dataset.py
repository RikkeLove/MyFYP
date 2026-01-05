import os
import glob
import random
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
from torch.utils.data import Dataset


def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _list_mat_files(root_dir: str) -> List[str]:
    root_dir = _abs(root_dir)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Dataset dir not found: {root_dir}")

    mats = glob.glob(os.path.join(root_dir, "**", "*.mat"), recursive=True)
    mats += glob.glob(os.path.join(root_dir, "**", "*.MAT"), recursive=True)
    mats = sorted(set(mats))

    if not mats:
        raise FileNotFoundError(
            "No .mat files found under:\n"
            f"  {root_dir}\n"
            f"Tips:\n"
            f"  - Check folder really contains .mat\n"
            f"  - If running from IDE, working directory may differ\n"
        )
    return mats


def _load_hsi_mat(mat_path: str) -> np.ndarray:
    """
    Return float32 HSI cube (H, W, 28) in [0,1]
    - CAVE key usually: 'data_slice' (often uint16)
    - KAIST key usually: 'HSI' (often float)
    """
    d = sio.loadmat(mat_path)

    key = None
    for k in ("data_slice", "HSI"):
        if k in d and isinstance(d[k], np.ndarray):
            key = k
            break

    if key is None:
        # fallback: pick a 3D array whose spectral dim looks like >=10
        candidates = []
        for k, v in d.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] >= 10:
                candidates.append((k, v))
        if not candidates:
            raise KeyError(f"[{mat_path}] Cannot find 3D HSI array key. Keys={list(d.keys())}")
        key, _ = max(candidates, key=lambda kv: kv[1].size)

    hsi = d[key]
    if hsi.ndim != 3:
        raise ValueError(f"[{mat_path}] key={key} is not 3D. shape={hsi.shape}")

    # handle (C,H,W) -> (H,W,C)
    if hsi.shape[0] == 28 and hsi.shape[-1] != 28:
        hsi = np.transpose(hsi, (1, 2, 0))

    hsi = np.asarray(hsi).astype(np.float32)

    # normalize if looks like 16-bit
    mx = float(np.max(hsi))
    if mx > 1.5:
        hsi = hsi / 65535.0

    hsi = np.clip(hsi, 0.0, 1.0)

    if hsi.shape[-1] != 28:
        raise ValueError(f"[{mat_path}] Expected 28 bands, got {hsi.shape}")
    return hsi


def _random_crop(hsi: np.ndarray, patch: int) -> np.ndarray:
    H, W, C = hsi.shape
    if H < patch or W < patch:
        raise ValueError(f"HSI smaller than patch: hsi={hsi.shape}, patch={patch}")
    top = random.randint(0, H - patch)
    left = random.randint(0, W - patch)
    return hsi[top:top + patch, left:left + patch, :]


def _rot_flip(p: np.ndarray, rot_k: int, flip_h: bool, flip_v: bool) -> np.ndarray:
    out = np.rot90(p, k=rot_k, axes=(0, 1)).copy()
    if flip_h:
        out = out[:, ::-1, :].copy()
    if flip_v:
        out = out[::-1, :, :].copy()
    return out


def _down_up_to_fixed(p: np.ndarray, scale: float, out_size: int) -> np.ndarray:
    """
    输出固定 out_size×out_size×C：
    - scale=1.0: 不变
    - scale<1.0: 先下采样到 round(out_size*scale)，再上采样回 out_size
    """
    if abs(scale - 1.0) < 1e-6:
        return p.astype(np.float32)

    x = torch.from_numpy(p).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    small = max(1, int(round(out_size * scale)))
    x_small = F.interpolate(x, size=(small, small), mode="bilinear", align_corners=False)
    x_back = F.interpolate(x_small, size=(out_size, out_size), mode="bilinear", align_corners=False)
    out = x_back.squeeze(0).permute(1, 2, 0).contiguous().numpy()
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class HSICubeDatasetBHWC(Dataset):
    """
    输出：torch.FloatTensor [H,W,C]（DataLoader 会拼成 [B,H,W,C]）
    数据增强：
      - rot: 0/90/180/270
      - flip: h / v
      - scale: 1 / 0.8 / 0.6（下采样再上采样回 256）
    """
    def __init__(self, cave_dir: str, kaist_dir: str,
                 patch_size: int = 256, length: int = 20000,
                 use_kaist_prob: float = 0.5,
                 enable_aug: bool = True,
                 cache_size: int = 2):
        super().__init__()
        self.cave_files = _list_mat_files(cave_dir)
        self.kaist_files = _list_mat_files(kaist_dir)

        self.patch_size = patch_size
        self.length = length
        self.use_kaist_prob = float(use_kaist_prob)
        self.enable_aug = enable_aug

        self.scales = (1.0, 0.8, 0.6)

        # tiny cache per worker process
        self.cache_size = max(0, int(cache_size))
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []

        print("[HSICubeDatasetBHWC]")
        print("  CAVE  mats:", len(self.cave_files))
        print("  KAIST mats:", len(self.kaist_files))

    def __len__(self):
        return self.length

    def _get_hsi(self, path: str) -> np.ndarray:
        if self.cache_size <= 0:
            return _load_hsi_mat(path)

        if path in self._cache:
            return self._cache[path]

        hsi = _load_hsi_mat(path)
        self._cache[path] = hsi
        self._cache_order.append(path)
        if len(self._cache_order) > self.cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return hsi

    def __getitem__(self, idx: int):
        use_kaist = (random.random() < self.use_kaist_prob)
        path = random.choice(self.kaist_files if use_kaist else self.cave_files)

        hsi = self._get_hsi(path)               # (H,W,28)
        patch = _random_crop(hsi, self.patch_size)

        if self.enable_aug:
            rot_k = random.randint(0, 3)        # 0/90/180/270
            flip_h = random.randint(0, 1) == 1
            flip_v = random.randint(0, 1) == 1
            patch = _rot_flip(patch, rot_k, flip_h, flip_v)

            sc = random.choice(self.scales)     # 1/0.8/0.6
            patch = _down_up_to_fixed(patch, sc, out_size=self.patch_size)

        return torch.from_numpy(patch).float()


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
