import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FleetwoodNet16V1(nn.Module):
    def __init__(self):
        super().__init__()
        # [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
