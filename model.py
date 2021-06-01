import torch
import torch.nn as nn
from prepare import PrepNetwork
from hide import HideNetwork
from reveal import RevealNetwork

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.m1 = PrepNetwork()
        self.m2 = HideNetwork()
        self.m3 = RevealNetwork()

    def forward(self, secret, cover):
        # TODO: return container and revealed for learning
        return
