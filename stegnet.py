import torch
import torch.nn as nn
from prepare import PrepNetwork
from hide import HidingNetwork
from reveal import RevealNetwork


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.m1 = PrepNetwork()
        self.m2 = HidingNetwork()
        self.m3 = RevealNetwork()

    def forward(self, secret, cover):
        prepped = self.m1(secret)
        mid = torch.cat((prepped, cover), 1)
        container, container_noise = self.m2(mid)
        revealed = self.m3(container_noise)
        return container, revealed
