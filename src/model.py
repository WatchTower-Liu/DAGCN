import torch
from torchvision import models
from torch import nn
from FPN import FPN
from VGG import VGG16
from AnwGCN import AnwGCN
from spaceADJgen import generateSADJ
from decoder import mainDecoder, edgeDecoder
from scaleInteraction import scaleInteraction
import config
import torch.nn.functional as F
from ptflops import get_model_complexity_info

class DAGCN(nn.Module):
    def __init__(self, inputChannel, outputChannel, batchSzie, aggNum):
        super().__init__()
        self._inputChannel = inputChannel
        self._outputChannel = outputChannel
        self._batchSzie = batchSzie
        self._aggNum = aggNum

        self.SADJG()
        self.build()

    def SADJG(self):
        # self._56SA = torch.tensor(generateSADJ(56, 100), dtype=torch.long).to(config.device)
        self._28SA = torch.tensor(generateSADJ(28, 100), dtype=torch.long).to(config.device)

    def build(self):
        self.backbone = VGG16(False)

        self.FPN = FPN()

        # self.GCN56_1 = nn.Sequential(
        #     AnwGCN(128, 128, 56, self._aggNum, "S", 100, self._56SA)
        # )
        # self.GCN56_2 = nn.Sequential(
        #     AnwGCN(128, 128, 56, self._aggNum, "S", 100, self._56SA)
        # )
        self.GCN28_1 = nn.Sequential(
            AnwGCN(256+128, 256, 28, self._aggNum, "S", 100, self._28SA), 
        )
        self.GCN28_2 = nn.Sequential(
            AnwGCN(256, 128, 28, self._aggNum, "S", 100, self._28SA)
        )
        self.GCN14_1 = nn.Sequential(
            AnwGCN(512, 256, 14, self._aggNum, "F")
        )
        self.GCN14_2 = nn.Sequential(
            AnwGCN(256, 128, 14, self._aggNum, "F")
        )

        self.MDecoder = mainDecoder([128, 128, 128, 128], self._outputChannel)
        self.EDecoder = edgeDecoder(128, self._outputChannel)

        self.norm_56 = nn.InstanceNorm2d(128)

    def getADJandDIS(self):
        return self.showADJ, self.showDIS

    def getADJandDISL2(self):
        return self.showADJL2, self.showDISL2

    def forward(self, X):
        useFeatureList = self.backbone(X)
        FPNR = self.FPN(*useFeatureList)  # FPN output size from small to big
        self.showADJ = []
        self.showDIS = []
        GCN1Out_1, adj14, dis14 = self.GCN14_1(FPNR[0])
        self.showADJ.append(adj14)
        self.showDIS.append(dis14)
        _56_in = F.interpolate(FPNR[2], (28, 28))
        # print(_56_in.shape)
        # print(FPNR[1].shape)
        GCN2Out_1, adj28, dis28 = self.GCN28_1(torch.cat([FPNR[1], _56_in], dim=1))
        self.showADJ.append(adj28)
        self.showDIS.append(dis28)
        # GCN3Out_1,_,_ 
        #interaction feature size from big to small
        # interactionF1, interactionF2, interactionF3 = self.inter([GCN3Out_1, GCN2Out_1, GCN1Out_1])
        self.showADJL2 = []
        self.showDISL2 = []
        GCN1Out, adj14, dis14 = self.GCN14_2(GCN1Out_1)
        self.showADJL2.append(adj14)
        self.showDISL2.append(dis14)
        GCN2Out, adj28, dis28 = self.GCN28_2(GCN2Out_1)
        self.showADJL2.append(adj28)
        self.showDISL2.append(dis28)
        # GCN3Out,_,_ = self.GCN56_2(GCN3Out_1)
        # self.showF = GCN3Out
        EOut = self.EDecoder(FPNR[3])
       
        _56_Out = FPNR[2] + F.interpolate(GCN2Out, (FPNR[2].shape[2], FPNR[2].shape[3]))
        SOut = self.MDecoder(GCN1Out, GCN2Out, _56_Out, FPNR[3])
        

        return SOut, EOut
            

def main():
    model = DAGCN(3, 1, 2, 8)
    # print(model)
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # inputData = torch.randn(1, 3, 224, 224)
    # model(inputData)

if __name__ == "__main__":
    main()
