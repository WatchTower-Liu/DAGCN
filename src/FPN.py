import torch
from torch import nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.build()

    def build(self):
        self.layer_1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.LeakyReLU(inplace = True)
        )
        self.layer_1_c = nn.Sequential(
            nn.Conv2d(512, 128, 3, 1, padding=1),
            nn.LeakyReLU(inplace = True)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, padding=1),
            nn.LeakyReLU(inplace = True)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, padding=1),
            nn.LeakyReLU(inplace = True)
        )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.LeakyReLU(inplace = True)
        )

    def forward(self, X1, X2, X3, X4):
        X1n = self.layer_1(X1)
        X1ns = F.interpolate(X1n, (X1n.shape[2] * 2, (X1n.shape[3] * 2)))
        X2n = self.layer_2(X2 + X1ns)
        X2ns = F.interpolate(X2n, (X2n.shape[2] * 2, (X2n.shape[3] * 2)))
        X3n = self.layer_3(X3 + X2ns)
        X3ns = F.interpolate(X3n, (X3n.shape[2] * 2, (X3n.shape[3] * 2)))
        X1ns2 = F.interpolate(X1n, (X1n.shape[2] * 8, (X1n.shape[3] * 8)))
        X4n = self.layer_4(X4 + X3ns + self.layer_1_c(X1ns2))
        
        return X1n, X2n, X3n, X4n

