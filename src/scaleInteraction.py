import torch 
from torch import nn
import torch.nn.functional as F


class scaleInteraction(nn.Module):
    def __init__(self, intercationChannel, intercationFeatureNum, fuseInedx):
        super().__init__()
        self._intercationChannel = intercationChannel
        self._intercationFeatureNum = intercationFeatureNum
        self._fuseInedx = fuseInedx

        self.build()

    def inputChannel(self, I):
        inputChannelNum = self._intercationChannel[I]+ \
        sum([self._intercationChannel[N] for N in self._fuseInedx[I]])
        return inputChannelNum

    def build(self):
        self.channelFuse = nn.ModuleList()
        for i in range(self._intercationFeatureNum):
            self.channelFuse.append(nn.Sequential(
                nn.Conv2d(self.inputChannel(i), 
                          self._intercationChannel[i], 1, 1),
                nn.LeakyReLU()
            ))

    def interaction(self, X, baseFI, interactionFI):
        baseSize = X[baseFI].shape
        fuseF = [X[baseFI]]
        for interactionIndex in interactionFI:
            iF = X[interactionIndex]
            fuseF.append(F.interpolate(iF, (baseSize[2], baseSize[3])))
        fuseFeature = torch.cat(fuseF, dim = 1)
        return self.channelFuse[baseFI](fuseFeature)

    def forward(self, X):
        if len(X) != self._intercationFeatureNum:
            raise ValueError("please set interaction feature num same as input feature num")
        returnF = []
        for i in range(self._intercationFeatureNum):
            returnF.append(self.interaction(X, i, self._fuseInedx[i]))

        return returnF
