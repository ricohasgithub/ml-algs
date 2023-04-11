
import torch
import torch.nn as nn

import numpy as np

class HMM:
    
    def __init__(self):
        pass

class Autoencoder(nn.Module):

    def __init__(self, decoder=None):
        super().__init__()

    def forward(self, x):
        z = self.encoder(x)
        x_p = self.decoder(z)
        return x_p