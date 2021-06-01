import torch.nn as nn


class HideNetwork(nn.Module):
    def __init__(self):
        super(HideNetwork, self).__init__()

    def forward(self, x):
        return x
