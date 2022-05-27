import torch
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

def make_layer(block, in_c, out_c, repeat, dilation=1, last=None, preact=True):
    stacked = []
    for i in range(repeat):
        if i==0:
            stacked.append(block(in_c, out_c, dilation=dilation, stride=1, preact=preact))
        else:
            stacked.append(block(out_c, out_c, dilation=1, stride=1, preact=preact))
    if last is not None:
        stacked.append(last)
    return nn.Sequential(*stacked)

def auto_calcul(num_c = '64,128,256,512', k=2):
    num_c = [*map(int,num_c.split(','))]
    in_added = [num_c[-1]]
    for i in range(3):
        in_added.append(int((in_added[i] + num_c[-i - 1]) / k))
    in_added.append(int((in_added[3] + num_c[0])))

    num_c.append(in_added[2] + num_c[1])
    num_c.append(in_added[1] + num_c[2])
    num_c.append(num_c[3])
    for i in range(4, 7):
        in_added.append(in_added[i] + num_c[i])
    return in_added, num_c[3::-1]+num_c[4:]
## blocks
class BN_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1, bottleneck=4, dilation=1, se=False, preact=True): #
        super().__init__()
        self.bottleneck = bottleneck
        self.bn_c = int(out_c/bottleneck)
        self.se = se
        self.preact = preact
        if preact:
            self.conv = nn.Sequential(nn.BatchNorm2d(in_c),
                                      nn.ReLU(),
                                      nn.Conv2d(in_c, self.bn_c, kernel_size=1, stride=1, bias=False),
                                      nn.BatchNorm2d(self.bn_c),
                                      nn.ReLU(),
                                      nn.Conv2d(self.bn_c, self.bn_c, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False),
                                      nn.BatchNorm2d(self.bn_c),
                                      nn.ReLU(),
                                      nn.Conv2d(self.bn_c, out_c, kernel_size=1, stride=1, bias=False))
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_c, self.bn_c, kernel_size=1, stride=1, bias=False),
                                      nn.BatchNorm2d(self.bn_c),
                                      nn.ReLU(),
                                      nn.Conv2d(self.bn_c, self.bn_c, kernel_size=3, stride=stride, padding=dilation,
                                                dilation=dilation, bias=False),
                                      nn.BatchNorm2d(self.bn_c),
                                      nn.ReLU(),
                                      nn.Conv2d(self.bn_c, out_c, kernel_size=1, stride=1, bias=False),
                                      nn.BatchNorm2d(out_c))

        if se:
            self.bridge = SE_block(out_c, out_c)

        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if (stride != 1) | (in_c != out_c):
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        if self.se:
            feature_map = self.conv(x)
            x = self.shortcut(x) + feature_map*self.bridge(feature_map)
        else:
            x = self.shortcut(x) + self.conv(x)
        if not self.preact:
            x = self.relu(x)
        return x

class SE_block(nn.Module):
    def __init__(self, in_c, out_c, reduction_rate=16):
        super().__init__()
        bn_c = int(in_c / reduction_rate)
        self.se_net = nn.Sequential(nn.AdaptiveAvgPool2d(1), # squeeze
                                    nn.Conv2d(in_c, bn_c, kernel_size=1, stride=1, bias=False),
                                    nn.ReLU(),
                                    nn.Conv2d(bn_c, out_c, kernel_size=1, stride=1, bias=False),
                                    nn.Sigmoid()) # excitation
    def forward(self,x):
        return self.se_net(x)

class UDR_block(nn.Module):
    def __init__(self, in_c=1024, k=2, phase='up', preact=True):
        super().__init__()
        self.phase=phase
        if phase=='up':
            self.M = BN_block(in_c, int(in_c/k), preact=preact).conv
        else:
            self.M = BN_block(in_c, in_c, preact=preact).conv
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.dwsample = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     squeeze())

    def forward(self,x):
        if self.phase=='up':
            x_new = self.M(x) + self.r_func(x)
            x_new = self.upsample(x_new)
        elif self.phase=='down':
            x_new = x + self.M(x)
            x_new = self.dwsample(x_new)
        elif self.phase=='last':
            x_new = x + self.M(x)
            x_new = self.avgpool(x_new)
        return x_new

    def r_func(self, x, k=2):
        """
        :param x: tensor {n,c,h,w}
        :param k: int
        :return: x with reduced dimension
        """
        _, c, h, w = x.size()
        x_ = x.contiguous().view(-1, c, h * w)
        x_r_ = nn.AvgPool1d(kernel_size=k, stride=k)(x_.permute(0, 2, 1))
        x_r = x_r_.permute(0, 2, 1).contiguous().view(-1, int(c / k), h, w)
        return x_r * k

class CAT_block(nn.Module):
    def __init__(self, in_c=512, add_c=512, dilation=1, repeat=1, k=2, phase='up', preact=True):
        super().__init__()
        #
        self.phase=phase
        self.regular = make_layer(BN_block, in_c, in_c, repeat=repeat, dilation=dilation, preact=preact)
        self.transfer = BN_block(add_c, add_c, preact=preact).conv
        self.sample = UDR_block(in_c+add_c, k=k, phase=phase, preact=preact)
    def forward(self,x,x_add):
        """
        :param x:
        :param x_add:
        :return: tuple (concatenated, sampled)
        """
        x = self.regular(x) # input
        x_add = self.transfer(x_add)
        x_new = concat(x,x_add)
        out = self.sample(x_new)
        if self.phase=='up':
            return x_new,out
        else:
            return out
