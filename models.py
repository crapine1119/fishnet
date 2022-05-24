import numpy as np
import cv2 as cv
#
import albumentations as A
from albumentations.pytorch import ToTensorV2
#
import torch
import timm
from torch import nn
## basic modules
def concat(a,b):
    """
    :param a: tensor {n,c1,h,w}
    :param b: tensor {n,c2,h,w}
    :return: torch.cat([a,b])
    """
    return torch.cat([a,b],dim=1)

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

class BN_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1, bottleneck=4, dilation=1):
        super().__init__()
        self.bottleneck = bottleneck
        self.bn_c = int(out_c/bottleneck)

        self.conv = nn.Sequential(nn.Conv2d(in_c, self.bn_c, kernel_size=1, stride=1, bias=False), # this can also be implemented
                                  nn.BatchNorm2d(self.bn_c),
                                  nn.ReLU(),
                                  nn.Conv2d(self.bn_c, self.bn_c, kernel_size=1, stride=stride, bias=False, dilation=dilation),
                                  nn.BatchNorm2d(self.bn_c),
                                  nn.ReLU(),
                                  nn.Conv2d(self.bn_c, out_c, kernel_size=1, stride=1, bias=False),
                                  nn.BatchNorm2d(out_c))
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if (stride != 1) | (in_c != out_c):
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
         x = self.relu(self.conv(x) + self.shortcut(x))
         return x

class SE_block(nn.Module):
    def __init__(self, in_c, out_c, reduction_rate=16):
        super().__init__()
        bn_c = int(in_c / reduction_rate)

        self.se_net = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Conv2d(in_c, bn_c, kernel_size=1, stride=1, bias=False),
                                    nn.ReLU(),
                                    nn.Conv2d(bn_c, out_c, kernel_size=1, stride=1, bias=False),
                                    nn.Sigmoid()
                                    )

        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False))

    def forward(self,x):
        return self.se_net(x)*self.shortcut(x)

def make_layer(block, in_c, out_c, repeat, last=None):
    stacked = []
    for i in range(repeat):
        if i==0:
            stacked.append(block(in_c, out_c, stride=1))
        else:
            stacked.append(block(out_c, out_c, stride=1))
    if last is not None:
        stacked.append(last)
    return nn.Sequential(*stacked)

class UR_block(nn.Module):
    def __init__(self, in_c=512, add_c=512, block=BN_block, k=2, repeat=1):
        super().__init__()
        #
        self.regular = make_layer(block, in_c, in_c, repeat=repeat)
        self.transfer = make_layer(block, add_c, add_c, repeat=1)
        self.out_c = in_c + add_c

        self.M = BN_block(self.out_c, int(self.out_c/k)).conv
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self,x,x_add):
        x = self.regular(x) # input
        x_add = self.transfer(x_add)
        x = concat(x,x_add)
        x = self.M(x) + self.r_func(x)
        return self.upsample(x)

    def r_func(self, x, k=2):
        """
        :param x: tensor {n,c,h,w}
        :param k: int
        :return: x with reduced dimension
        """
        _, c, h, w = x.size()
        x_r = x.contiguous().view(-1, c, h * w)
        x_r = nn.AvgPool1d(kernel_size=k, stride=k)(x_r.permute(0, 2, 1))
        x_r = x_r.permute(0, 2, 1).contiguous().view(-1, int(c / k), h, w)
        return x_r * k

#class DR_block(nn.Module):
    # TODO

##


class tail(nn.Module):
    def __init__(self, in_c=3, block=BN_block, Ls='3,4,6,3'):
        super().__init__()
        self.in_c = in_c
        self.Ls = [*map(int,Ls.split(','))]
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.t1 = make_layer(block,  64, 128, repeat=self.Ls[0], last=nn.MaxPool2d(kernel_size=2,stride=2,padding=0))
        self.t2 = make_layer(block, 128, 256, repeat=self.Ls[1], last=nn.MaxPool2d(kernel_size=2,stride=2,padding=0))
        self.t3 = make_layer(block, 256, 512, repeat=self.Ls[2], last=nn.MaxPool2d(kernel_size=2,stride=2,padding=0))
        self.t4 = make_layer(block, 512,1024, repeat=self.Ls[3], last=SE_block(1024, 512, reduction_rate=16))

    def forward(self,x):
        x1 = self.conv1(x)  # x1 : n,64,56,56 in paper
        x2 = self.t1(x1)
        x3 = self.t2(x2)
        x4 = self.t3(x3)
        to_body = self.t4(x4)
        return {'1': x1,    '2': x2,    '3': x3,    '4': x4,    'to_body': to_body}

class body(nn.Module):
    def __init__(self, in_c, k=2, block=BN_block, Ls='1,1,1'):
        super().__init__()
        self.in_c = in_c
        self.Ls = [*map(int,Ls.split(','))]


        self.b4 = UR_block(in_c,            in_c,        block, k=k, repeat=self.Ls[0])  # 512 512
        self.b3 = UR_block(in_c,            int(in_c/2), block, k=k, repeat=self.Ls[1])  # 512 256
        self.b2 = UR_block(int(in_c*3/4),   int(in_c/4),  block, k=k, repeat=self.Ls[2]) # 384 128
        # b1은 up되지 않기 때문에 DR에서 구현

    def forward(self, x):
        x4 = self.b4(x['to_body'],x['4'])
        x3 = self.b3(x4,x['3'])
        x2 = self.b2(x3,x['2'])
        return {'1':x['1'], '2':x2, '3':x3, '4':x4}

"""
net1 = tail()
net2 = body(in_c=512)
out = net1(img)
out2 = net2(out)
for i in out2:
    print(out2[i].shape)
"""

#class head(nn.Module):

class fishnet(nn.Module):
    def __init__(self, in_c=3, out_c=10, block=BN_block, Ls='3,4,6,3'):
        super().__init__()

        self.tail = tail(in_c, block, Ls)
        self.bridge   = SE_block(2048, reduction_rate=16)


    def forward(self,x):
        mat = self.tail(x)
        to_head = mat['4']
        #
        mat = self.body(mat)
        mat['4'] = to_head
        #
        x = self.head_forward(mat)
        return x

    def body_forward(self,x):
        x4 = self.bridge(x['out'])

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

