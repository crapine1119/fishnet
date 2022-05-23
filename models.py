import numpy as np
import cv2 as cv
#
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
#
import torch
import timm
from torch import nn
##
class squeeze(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x.squeeze()

class unsqueeze(nn.Module):
    def __init__(self, dim=-1, n=1):
        super().__init__()
        self.n = n
        self.dim = dim
    def forward(self,x):
        for _ in range(self.n):
            x = x.unsqueeze(self.dim)
        return x

class bottleneck(nn.Module):
    def __init__(self, in_c, out_c, stride=2, bottleneck=4):
        super().__init__()
        self.bottleneck = bottleneck
        self.bn_c = int(out_c/bottleneck)

        self.conv = nn.Sequential(nn.Conv2d(in_c, self.bn_c, kernel_size=1, stride=1, bias=False), # this can also be implemented
                                  nn.BatchNorm2d(self.bn_c),
                                  nn.ReLU(),
                                  nn.Conv2d(self.bn_c, self.bn_c, kernel_size=1, stride=stride, bias=False),
                                  nn.BatchNorm2d(self.bn_c),
                                  nn.ReLU(),
                                  nn.Conv2d(self.bn_c, out_c, kernel_size=1, stride=1, bias=False),
                                  nn.BatchNorm2d(out_c),
                                  nn.ReLU())

        self.shortcut = nn.Sequential()
        if (stride != 1) | (in_c != out_c):
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
         x = self.conv(x) + self.shortcut(x)
         return x

class se_net(nn.Module):
    def __init__(self, in_c, reduction_rate=16):
        super().__init__()
        bn_c = int(in_c / reduction_rate)
        self.se_net = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    squeeze(),
                                    nn.Linear(in_c, bn_c, bias=False), nn.ReLU(),
                                    nn.Linear(bn_c, in_c, bias=False), nn.Sigmoid(),
                                    unsqueeze(dim=-1,n=2))

    def forward(self,Uc):
        Sc = self.se_net(Uc)
        return Sc*Uc

class fishnet(nn.Module):
    def __init__(self, in_c=3, out_c=10, block=bottleneck, Ls='3,4,6,3'):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.Ls = [*map(int,Ls.split(','))]
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self.make_layer(block,  64, 256, self.Ls[0])
        self.layer2 = self.make_layer(block, 256, 512, self.Ls[1])
        self.layer3 = self.make_layer(block, 512,1024, self.Ls[2])
        self.layer4 = self.make_layer(block, 1024,2048, self.Ls[3])
        self.bridge   = se_net(2048, reduction_rate=16)

    def make_layer(self, block, in_c, out_c, repeat):
        stacked = []
        for i in range(repeat):
            if i==0:
                stacked.append(block(in_c, out_c, stride=2))
            else:
                stacked.append(block(out_c, out_c, stride=1))
        return nn.Sequential(*stacked)

    def forward(self,x):
        x = self.tail_forward(x)
        x = self.body_forward(x)
        x = self.head_forward(x)
        return x

    def tail_forward(self,x):
        x1 = self.conv1(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        to_body = self.layer4(x4)
        return {'1': x1,    '2': x2,    '3': x3,    '4': x4,    'out': to_body}

    def body_forward(self,x):
        # TODO
        return
    def head_forward(self,x):
        # TODO
        return

"""
res50 = fishnet(Ls='3,4,6,3')
res101 = fishnet(Ls='3,4,23,3')
res152 = fishnet(Ls='3,8,36,3')
"""