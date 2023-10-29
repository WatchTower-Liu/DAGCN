import torch 
from torch import nn, sub
from torch._C import Value 
import torch.nn.functional as F
from util import _logmap, _expmap, _mobius_matvec

class SRKNN(nn.Module):
    ESP= 1e-10
    def __init__(self, K, shape, inputChannel, Ktype, SK=None, spaceADJ=None) -> None:
        """
        shape of spaceADJ is a vector(1D)
        """
        super().__init__()
        self._K = K
        self._shape = shape
        self._inputChannel = inputChannel
        self._Ktype = Ktype
        self._SK = SK
        self._spaceADJ = spaceADJ

        self.build()

    def build(self):
        self.minuendW = nn.Linear(self._inputChannel, self._inputChannel//2, bias = False)
        self.subtrahendW = nn.Linear(self._inputChannel, self._inputChannel//2, bias = False)

        _M = torch.randn([self._inputChannel//2, self._inputChannel//2])
        self.M = nn.Parameter(_M)

    def log_map(self, X):
        return _logmap(X)

    def exp_map(self, X):
        return _expmap(X)

    def forward(self, X: torch.Tensor):
        """
        input X: just need feature map.
        """
        X = X.permute(0, 2, 3, 1)    # convert to channel last
        Xshape = X.shape
        XV = X.reshape(Xshape[0], -1, Xshape[-1])
        # XV = self.exp_map(XV)
        nodeNum = XV.shape[1]
        if self._Ktype == "F":
            minuend = torch.reshape(XV.repeat(1, 1, nodeNum), (Xshape[0], -1, Xshape[-1]))
            subtrahend = XV.repeat(1, nodeNum, 1)
        elif self._Ktype == "S":
            if self._SK is None and self._spaceADJ is None:
                raise ValueError("place set space attr")
            minuend = torch.reshape(XV.repeat(1, 1, self._SK), (Xshape[0], -1, Xshape[-1]))
            # print(self._spaceADJ.shape)
            subtrahend = torch.index_select(XV, 1, self._spaceADJ)   #all image in this size have same spaceADJ
        else:
            raise ValueError("no such type of SRKNN")
        
        
        # print(subtrahend)
        subR = self.log_map(self.minuendW(minuend)) - self.log_map(self.subtrahendW(subtrahend))
        subR = F.sigmoid(subR)
        A = self.M.T@self.M
        dis = torch.sqrt(torch.sum(torch.einsum("ijk,kl->ijl", subR, A)*subR, dim=2))
        if self._Ktype == "F":
            diff = torch.neg(dis).reshape(Xshape[0], nodeNum, nodeNum)
        elif self._Ktype == "S":
            diff = torch.neg(dis).reshape(Xshape[0], nodeNum, self._SK)
        else:
            raise ValueError("no such type of SRKNN")
        diff = F.sigmoid(diff)
        value, index = torch.topk(diff, self._K, sorted = False)
        value = torch.neg(value.reshape(Xshape[0], -1, 1))
        if self._Ktype == "F":
            index = index.reshape(Xshape[0], -1)
            return index, value
        elif self._Ktype == "S":
            if not hasattr(self, "gather1DIndex"):
                gather1DIndex = torch.range(0, nodeNum-1).reshape(-1, 1).repeat(1, self._K).reshape(-1).long()
                setattr(self, "gather1DIndex", gather1DIndex)
            newADJ = []
            spaceA = self._spaceADJ.reshape(nodeNum, self._SK)
            for i in range(Xshape[0]):
                newADJ.append(spaceA[self.gather1DIndex, index[i].reshape(-1)])
            newADJ = torch.stack(newADJ, 0).long()
            return newADJ, value
        else:
            raise ValueError("no such type of SRKNN")

def main():
    spaceadj = torch.ones(400).long()
    tK = SRKNN(2, 2, "S", 4, spaceadj)
    data = torch.range(1, 400).reshape(2, 2, 10, 10)
    tK(data)


if __name__ == "__main__":
    main()
