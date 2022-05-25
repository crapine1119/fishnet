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
         return self.relu(self.conv(x) + self.shortcut(x))

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

class UDR_block(nn.Module):
    def __init__(self, in_c=1024, block=BN_block, k=2, phase='up'):
        super().__init__()
        self.phase=phase
        if phase=='up':
            self.M = block(in_c, int(in_c/k)).conv
        else:
            self.M = block(in_c, in_c).conv
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
    def __init__(self, in_c=512, add_c=512, block=BN_block, repeat=1, k=2, phase='up'):
        super().__init__()
        #
        self.phase=phase
        self.regular = make_layer(block, in_c, in_c, repeat=repeat)
        self.transfer = make_layer(block, add_c, add_c, repeat=repeat)
        self.sample = UDR_block(in_c+add_c, BN_block, k=k, phase=phase)
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
##
class tail(nn.Module):
    def __init__(self, in_c=3, block=BN_block, Ls='3,4,6,3', first_repeat=2):
        super().__init__()
        self.in_c = in_c
        self.Ls = [*map(int,Ls.split(','))]

        self.conv1 = make_layer(block, in_c, 64, repeat=first_repeat, last=nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # self.conv1 = nn.Sequential(nn.Conv2d(in_c, 64, kernel_size=, stride=2, padding=3, bias=False),
        #                            nn.BatchNorm2d(64),
        #                            nn.ReLU(),
        #                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
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
        return {'1': x1,    '2': x2,    '3': x3,    '4': x4,    'bridge': to_body}

class body_head(nn.Module):
    def __init__(self, in_c, k=2, block=BN_block, Ls_body='1,1,1', Ls_head='1,1,1,1'):
        super().__init__()
        self.in_c = in_c
        self.Ls_body = [*map(int,Ls_body.split(','))]
        self.Ls_head = [*map(int,Ls_head.split(','))]

        # body
        self.b4 = CAT_block(in_c,           in_c,           block,  repeat=self.Ls_body[0], phase='up', k=k)  # eg. 512 + 512 -> 1024
        self.b3 = CAT_block(in_c,           int(in_c/2),    block,  repeat=self.Ls_body[1], phase='up', k=k)  # 512 + 256
        self.b2 = CAT_block(int(in_c*3/4),  int(in_c/4),    block,  repeat=self.Ls_body[2], phase='up', k=k) # 384 + 128
        self.h1 = CAT_block(int(in_c*1/2),  int(in_c/8),    block,  repeat=self.Ls_head[0], phase='down', k=k)
        self.h2 = CAT_block(int(in_c*5/8),  in_c,           block,  repeat=self.Ls_head[1], phase='down', k=k)
        self.h3 = CAT_block(int(in_c*13/8), int(in_c*3/2),  block,  repeat=self.Ls_head[2], phase='down', k=k)
        self.h4 = CAT_block(int(in_c*25/8), in_c,           block,  repeat=self.Ls_head[3], phase='last', k=k)

    def forward(self, x):
        _ , x3_ = self.b4(x['bridge'], x['4'])
        x3, x2_ = self.b3(x3_, x['3']) # cated, sampled
        x2, out =self.b2(x2_, x['2'])
        #
        out = self.h1(out,  x['1'])
        out = self.h2(out, x2)
        out = self.h3(out, x3)
        out = self.h4(out, x['4'])
        return out

class fishnet(nn.Module):
    def __init__(self, in_c=3, out_c=10, k=2, Ls_tail='3,4,6,3', Ls_body='1,1,1', Ls_head='1,1,1,1', block=BN_block, first_repeat=2):
        super().__init__()

        self.tail = tail(in_c, block, Ls_tail, first_repeat=first_repeat)
        self.body_head = body_head(in_c=512, k=k, block=BN_block, Ls_body=Ls_body, Ls_head=Ls_head)
        self.cls = nn.Linear(2112,out_c, bias=False)

    def forward(self,x):
        x = self.tail(x)
        x = self.body_head(x)
        x = self.cls(x)
        return x

def make_fish(**kwargs):
    return fishnet(**kwargs)

"""
* Fish100 
cfg = {'first_repeat':1,
       'k':2,
       'Ls_tail':'3,4,6,3',
       'Ls_body':'1,1,1',
       'Ls_head':'1,1,1,1'}
fish100 = make_fish(**cfg)
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
3  : first res block (2 in paper)
55 : tail; 3463 *3(resnet) + 4(shortcut) + 3(SE block)
42 : 7(body ~ head)*2(raw & added)*3(res block)
total 100 layers
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

* Fish150
cfg = {'first_repeat':2,
       'k':2,
       'Ls_tail':'3,4,6,3',
       'Ls_body':'1,2,2',
       'Ls_head':'2,2,2,4'}
fish150 = make_fish(**cfg)
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
6  : first res block (2 in paper)
55 : tail; 3463 *3(resnet) + 4(shortcut) + 3(SE block)
90 : 15(body ~ head)*2(raw & added)*3(res block)
total 151 layers
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
"""