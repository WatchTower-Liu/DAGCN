import torch
from torch import nn

from torchvision import models

class VGG16(nn.Module):
    useFeatureNum = [8, 15, 22, 29]
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.vgg16(pretrained=pretrained).features
        self.layerNum = len(list(self.backbone.named_children()))
        # print(len(list(self.backbone.named_children())))

    def forward(self, X):
        useFeature = []
        for i in range(self.layerNum):
            X = self.backbone[i](X)
            if i in self.useFeatureNum:
                useFeature.append(X)
        return useFeature[::-1]

def main():
    B = VGG16(False)
    data = torch.randn(2, 3, 224, 224)
    d = B(data)
    for dd in d:
        print(dd.shape)

if __name__ == "__main__":
    main()
