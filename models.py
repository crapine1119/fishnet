from fish_blocks import *
##
class tail(nn.Module):
    def __init__(self, in_c=3, out_c=10, Ls='3,4,6,3', num_c='64,128,256,512', preact=True, model='fishnet'):
        super().__init__()
        self.in_c = in_c
        self.Ls = [*map(int,Ls.split(','))]
        self.num_c = [*map(int,num_c.split(','))]
        self.model = model
        if preact:
            self.conv1 = nn.Sequential(nn.Conv2d(in_c, self.num_c[0], kernel_size=3, stride=1, padding=1, bias=False),
                                       BN_block(self.num_c[0], self.num_c[0], preact=preact).conv,
                                       BN_block(self.num_c[0], self.num_c[0], preact=preact).conv)

        else:
          self.conv1 = nn.Sequential(BN_block(in_c,         self.num_c[0], preact=preact).conv,
                                     BN_block(self.num_c[0],self.num_c[0], preact=preact).conv)

        self.t1 = make_layer(BN_block, self.num_c[0], self.num_c[1], repeat=self.Ls[0], preact=preact, last=nn.MaxPool2d(kernel_size=2,stride=2,padding=0))
        self.t2 = make_layer(BN_block, self.num_c[1], self.num_c[2], repeat=self.Ls[1], preact=preact, last=nn.MaxPool2d(kernel_size=2,stride=2,padding=0))
        self.t3 = make_layer(BN_block, self.num_c[2], self.num_c[3], repeat=self.Ls[2], preact=preact, last=nn.MaxPool2d(kernel_size=2,stride=2,padding=0))
        if model=='fishnet':
            self.t4 = make_layer(BN_block, self.num_c[3], self.num_c[3], repeat=self.Ls[3], preact=preact, last=BN_block(self.num_c[3], self.num_c[3], se=True, preact=preact))
        else: # resnet
            self.t4 = make_layer(BN_block, self.num_c[3], self.num_c[3], repeat=self.Ls[3], preact=preact, last=nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                                                                                                              squeeze(),
                                                                                                                              nn.Linear(self.num_c[3], out_c)))
    def forward(self,x):
        x1 = self.conv1(x)  # x1 : n,64,56,56 in paper
        x2 = self.t1(x1)
        x3 = self.t2(x2)
        x4 = self.t3(x3)
        to_body = self.t4(x4)
        if self.model=='fishnet':
            return {'1': x1,    '2': x2,    '3': x3,    '4': x4,    'bridge': to_body}
        else: #resnet
            return to_body

class body_head(nn.Module):
    def __init__(self, k=2, Ls_body='1,1,1', Ls_head='1,1,1,1', num_c = '64,128,256,512', preact=True):
        super().__init__()
        self.Ls_body = [*map(int,Ls_body.split(','))]
        self.Ls_head = [*map(int,Ls_head.split(','))]
        self.in_c, self.added_c = auto_calcul(num_c, k)

        # body
        self.b4 = CAT_block(self.in_c[0], self.added_c[0], repeat=self.Ls_body[0], phase='up', k=k, preact=preact)  # eg. 512 + 512 -> 1024
        self.b3 = CAT_block(self.in_c[1], self.added_c[1], repeat=self.Ls_body[1], phase='up', k=k, preact=preact, dilation=2)  # 512 + 256
        self.b2 = CAT_block(self.in_c[2], self.added_c[2], repeat=self.Ls_body[2], phase='up', k=k, preact=preact, dilation=4) # 384 + 128
        # head
        self.h1 = CAT_block(self.in_c[3], self.added_c[3], repeat=self.Ls_head[0], phase='down', k=k, preact=preact)
        self.h2 = CAT_block(self.in_c[4], self.added_c[4], repeat=self.Ls_head[1], phase='down', k=k, preact=preact)
        self.h3 = CAT_block(self.in_c[5], self.added_c[5], repeat=self.Ls_head[2], phase='down', k=k, preact=preact)
        self.h4 = CAT_block(self.in_c[6], self.added_c[6], repeat=self.Ls_head[3], phase='last', k=k, preact=preact)

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
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        final_c,_ = auto_calcul(self.hparams.num_c, self.hparams.k)
        self.tail = tail(self.hparams.in_c, Ls = self.hparams.Ls_tail, num_c = self.hparams.num_c, preact=self.hparams.preact, model=self.hparams.model)
        if self.hparams.model=='fishnet':
            self.body_head = body_head(k=self.hparams.k, Ls_body=self.hparams.Ls_body, Ls_head=self.hparams.Ls_head, num_c=self.hparams.num_c, preact=self.hparams.preact)
            self.cls = nn.Linear(final_c[-1], self.hparams.out_c, bias=False)

    def forward(self,x):
        x = self.tail(x)
        if self.hparams.model=='fishnet':
            x = self.body_head(x)
            x = self.cls(x)
        return x

def make_fish(hparams):
    return fishnet(hparams)