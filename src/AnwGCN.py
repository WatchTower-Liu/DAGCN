import torch 
from torch import nn
from SRKNN import SRKNN
from graph_conv import GraphConv


class AnwGCN(nn.Module):
    def __init__(self, inputChannel, outputChannel, shape, aggNum, graphBuildType, SK = None, SADJ = None):
        super().__init__()
        self._inputChannel = inputChannel
        self._outputChannel = outputChannel
        self._shape = shape
        self._aggNum = aggNum
        self._graphBuildType = graphBuildType
        self._SK = SK
        self._SADJ = SADJ

        self.build()

    def build(self):
        self.graphBuild = SRKNN(self._aggNum, 
                                self._shape, 
                                self._inputChannel, 
                                self._graphBuildType, 
                                self._SK,
                                self._SADJ)

        self.GCN = GraphConv(self._inputChannel, 
                             self._outputChannel, 
                             self._aggNum)

    def forward(self, X):
        self.ADJ, self.Dis = self.graphBuild(X)
        outFeature = self.GCN(X, self.ADJ, self.Dis)

        return outFeature, self.ADJ.detach().cpu().numpy(), self.Dis.detach().cpu().numpy()

    def getADJandDIS(self):
        return self.ADJ.detach().cpu().numpy(), self.Dis.detach().cpu().numpy()
