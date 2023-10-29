from platform import node
import torch 
from torch import nn
import numpy 
import torch.nn.functional as F
from util import _logmap, _expmap, _mobius_matvec

class GraphConv(nn.Module):
    ESP = 1e-10
    def __init__(self, inputChannel, outputChannel, aggNum) -> None:
        super().__init__()
        self._inputChannel = inputChannel
        self._outputChannel = outputChannel
        self._aggNum = aggNum

        self.build()

    def build(self):
        self.edge1W = nn.Linear(self._inputChannel, 1, bias=False)
        self.edge2W = nn.Linear(self._inputChannel, 1, bias=False)

        self.GConv = nn.Sequential(
            nn.Conv1d(self._inputChannel+1, self._outputChannel, self._aggNum, self._aggNum),
            nn.LeakyReLU(inplace = True)
        )

        self.CConv = nn.Sequential(
            nn.Linear(self._inputChannel, self._outputChannel, bias=False), 
            nn.LeakyReLU(inplace = True)
        )
        self.OutFuse = nn.Sequential(
            nn.Conv1d(self._outputChannel*2, self._outputChannel, 1, 1),
            nn.LeakyReLU(inplace = True)
        )

    def batch_gather(self, X, index):
        res = []
        for i in range(X.shape[0]):
            res.append(X[i, index[i], :])
        res = torch.stack(res, dim=0)
        return res

    def log_map(self, X):
        return _logmap(X)

    def exp_map(self, X):
        return _expmap(X)

    def forward(self, X:torch.Tensor, ADJ:torch.Tensor, DIS:torch.Tensor):
        """
        X is feature map;
        ADJ shape is [batch, nodenum*aggNum]
        DIS shape is [batch, nodenum*aggNum, 1]
        """ 
        VX = X.permute(0, 2, 3, 1).reshape(X.shape[0], -1, X.shape[1])
        Xshape = VX.shape   # shape is [batch, nodeNum, featureChannel]
        nodeFeaute = self.batch_gather(VX, ADJ)
        # print(nodeFeaute.shape)

        neighboor = self.log_map(self.edge1W(nodeFeaute))*(1/(DIS+self.ESP))
        center = self.log_map(self.edge2W(VX)).repeat(1, 1, self._aggNum).reshape(Xshape[0], -1, 1)*(1/(DIS+self.ESP))
        hyperEdge = F.sigmoid(neighboor+center)

        nodeFeaute = nodeFeaute * hyperEdge
        nodeFeaute = torch.cat([nodeFeaute, hyperEdge], dim=-1)
        Gconvr = self.GConv(nodeFeaute.permute(0, 2, 1))
        Cconvr = self.CConv(VX).permute(0, 2, 1)
        out = torch.cat([Gconvr, Cconvr], dim = 1)
        out = self.OutFuse(out).reshape(Xshape[0], self._outputChannel, X.shape[2], X.shape[3])
        
        return out


def main():
    data = torch.ones(2, 10, 14, 14)
    adj = torch.ones(2, 14*14*5).long()
    dis = torch.ones(2, 14*14*5, 1)
    testC = GraphConv(10, 5, 5)
    print(testC(data, adj, dis).shape)


if __name__ == "__main__":
    main()
