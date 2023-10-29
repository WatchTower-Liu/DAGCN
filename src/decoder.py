import torch 
from torch import nn
import torch.nn.functional as F

class mainDecoder(nn.Module):
    def __init__(self, hiddenChannelList, outputChannel):
        super().__init__()
        if not isinstance(hiddenChannelList, list):
            raise ValueError("config hidden layer channel must be list")
        self._hiddenChannelList = hiddenChannelList
        self._outputChannel = outputChannel

        self.build()

    def build(self):
        self.level_1 = nn.Sequential(
            nn.Conv2d(self._hiddenChannelList[0], 128, 3, 1, padding = 1),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(128, 64, 3, 1, padding = 1),
            nn.LeakyReLU(inplace = True)
        )
        self.level_2 = nn.Sequential(
            nn.Conv2d(self._hiddenChannelList[1]+64, 128, 3, 1, padding = 1),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(128, 64, 3, 1, padding = 1),
            nn.LeakyReLU(inplace = True)
        )
        self.level_3 = nn.Sequential(
            nn.Conv2d(self._hiddenChannelList[2]+64, 128, 3, 1, padding = 1),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(128, 64, 3, 1, padding = 1),
            nn.LeakyReLU(inplace = True)
        )
        self.level_4 = nn.Sequential(
            nn.Conv2d(self._hiddenChannelList[3]+64, 128, 3, 1, padding = 1),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(128, 64, 3, 1, padding = 1),
            nn.LeakyReLU(inplace = True)
        )
        self.level_5 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, padding = 1),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(32, self._outputChannel, 1, 1),
            nn.Softmax(dim = 1)
        )

    def forward(self, level1X, level2X, level3X, level4X):
        level1XO = self.level_1(level1X)
        level1XO = F.interpolate(level1XO, (28, 28))
        level2XO = self.level_2(torch.cat([level2X, level1XO], dim = 1))
        level2XO = F.interpolate(level2XO, (56, 56))
        level3XO = self.level_3(torch.cat([level3X, level2XO], dim = 1))
        level3XO = F.interpolate(level3XO, (112, 112))
        level4XO = self.level_4(torch.cat([level4X, level3XO], dim = 1))
        level4XO = F.interpolate(level4XO, (224, 224))
        level5XO = self.level_5(level4XO)

        return level5XO

class edgeDecoder(nn.Module):
    def __init__(self, inputChannel, outputChannel):
        super().__init__()
        self._inputChannel = inputChannel
        self._outputChannel = outputChannel

        self.build()

    def build(self):
        self.level_1 = nn.Sequential(
            nn.Conv2d(self._inputChannel, 128, 3, 1, padding = 1),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(128, 64, 3, 1, padding = 1),
            nn.LeakyReLU(inplace = True)
        )
        self.level_2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, padding = 1),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(32, self._outputChannel, 1, 1),
            nn.Softmax(dim = 1)
        )

    def forward(self, X):
        level1XO = self.level_1(X)
        level1XO = F.interpolate(level1XO, (224, 224))
        level2XO = self.level_2(level1XO)

        return level2XO


def main():
    pass

if __name__ == "__main__":
    main()