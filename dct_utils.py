import torch
import torch.nn as nn
import numpy as np
from scipy.fftpack import dct

class DCTlayer(nn.Module):
    """
    DCT Layer：对输入张量的指定维度（height 或 width）进行 DCT 变换。
    """