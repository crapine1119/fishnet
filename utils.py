import numpy as np
import pandas as pd
import cv2 as cv
import pickle
import random
import os
import matplotlib.pyplot as plt
from glob import glob as glob
#
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
#
import torch
import timm
from torch import nn
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import torch.optim.lr_scheduler as lr_scheduler
#
from pytorch_lightning.loggers import TestTubeLogger as Tube
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning import LightningModule
from sklearn.metrics import f1_score, accuracy_score
#
from models import *
from tqdm import tqdm as tqdm
from ptflops import get_model_complexity_info
# https://github.com/sovrasov/flops-counter.pytorch
## preprocess
def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print('Seed : %s'%seed)

def unpickle(file):
    # http://www.cs.toronto.edu/~kriz/cifar.html
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict

def check_img(rdir):
    files = glob('%s/*data_batch*'%rdir)
    assert len(files)>0, 'No file error'
    d = unpickle(files[0])
    imgset_flat = d[b'data']
    imgset = imgset_flat.reshape(-1, 3, 32, 32)
    batch = torch.LongTensor(imgset[:64])
    plt.figure(figsize=[18,8])
    plt.imshow(make_grid(batch, padding=2).permute(1, 2, 0).numpy())

def load_cifar(rdir):
    trn_files = glob('%s/*data_batch*'%rdir)
    assert len(trn_files)>0, 'No file error'
    tst_file = '%s/test_batch'%rdir
    # train set
    trnx,trny = [],[]
    for fnm in trn_files:
        batch = unpickle(fnm)
        trnx.append(batch[b'data'])
        trny.extend(batch[b'labels'])
    trnx = np.concatenate(trnx).reshape(-1,3,32,32)
    trny = np.array(trny)

    # test set
    batch = unpickle(tst_file)
    tstx = batch[b'data'].reshape(-1,3,32,32)
    tsty = np.array(batch[b'labels'])
    return trnx,tstx,trny,tsty

class custom(Dataset):
    def __init__(self, imgs, labels, trans):
        super().__init__()
        self.imgs =imgs
        self.labels = labels
        self.trans = trans

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img = self.imgs[item]
        img = np.transpose(img, (1,2,0))
        label = self.labels[item]
        mat = {}
        img = self.trans(image=img)['image']
        mat['img'] = torch.FloatTensor(img)
        mat['label'] = torch.LongTensor([label])
        return mat

def get_params(model, input_size):
    with torch.cuda.device(0):
      macs, params = get_model_complexity_info(model, input_size, as_strings=False,
                                               print_per_layer_stat=False, verbose=True)

    print('FLOP : %2.2f G'%(2*macs*1e-9))
    print('Params : %2.2f M'%(params*1e-6))

def load_ckp(n=1, path=r'C:\Users\82109\PycharmProjects\fishnet\log',fig=True):
    if fig:
        fig = plt.figure(figsize=[18,6])
        ax1_leg,ax2_leg = [],[]

        pth_lst = sorted(glob(r'%s\*/*/*.ckpt' % path), key=lambda x:int(x.split('\\')[-3][8:]),reverse=True)

        for i in range(n):
            best_pth = pth_lst[i]
            version = best_pth.split('\\')[-3][8:]
            result = pd.read_csv(r'%s/../../metrics.csv'%best_pth)
            ax1 = fig.add_subplot(121);ax2 = fig.add_subplot(122)
            #
            grouped1 = result.groupby('epoch').mean()
            grouped1.plot(y=['trn_loss'],ax=ax1, linestyle='--', marker='.', markersize=5, alpha=0.3)
            grouped1.plot(y=['val_loss'],ax=ax1, linestyle='-', marker='.', markersize=5, alpha=0.8)
            ax1_leg.extend(['trn_loss(%s)' % version, 'val_loss(%s)' % version])
            ax1.set_ylim(0,5)
            #
            grouped2 = result.groupby('epoch').mean()
            grouped2.plot(y=['trn_f1'],ax=ax2, linestyle='--', marker='*', markersize=5, alpha=0.3)
            grouped2.plot(y=['val_f1'],ax=ax2, linestyle='-', marker='*', markersize=5, alpha=0.8)
            ax2_leg.extend(['trn_f1(%s)' % version, 'val_f1(%s)' % version])
            ax2.set_ylim(0, 1)
        ax1.grid('on')
        ax1.axhline(0,color='r')
        ax2.grid('on')
        ax2.axhline(1,color='r')
        ax1.legend(ax1_leg)
        ax2.legend(ax2_leg)
    return glob(r'%s\*/*/*.ckpt'%path)[-1]

def get_err(model, hparams, ckp, tst_loader):
    model = model.load_from_checkpoint(checkpoint_path=ckp, hparams=hparams)
    model.cuda()
    model.eval()
    num1,numk=0,0
    n_dset = 0
    for i in tqdm(tst_loader):
        with torch.no_grad():
            ans = model(i['img'].cuda())
        _,top1 = ans.topk(1,dim=-1)
        _,top5 = ans.topk(5,dim=-1)
        num1+=(top1.cpu()==i['label']).squeeze().sum().item()
        numk+=(top5.cpu()==i['label']).any(dim=1).sum().item()
        n_dset+=len(top1)
    return {'top1':num1/n_dset,
            'top5':numk/n_dset}

### pl
def get_callbacks(hparams):
    log_path = '%s'%(hparams.sdir)
    tube     = Tube(name=hparams.tube_name, save_dir=log_path)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          filename='{epoch:02d}-{val_loss:.4f}',
                                          save_top_k=1,
                                          mode='min')
    early_stopping        = EarlyStopping(monitor='val_loss',
                                          patience=hparams.es_patience,
                                          verbose=True,
                                          mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    print('get (1)ckp, (2)es, (3)lr_monitor callbacks with (4)tube')
    return {'callbacks':[checkpoint_callback, early_stopping, lr_monitor],
            'tube':tube}

class net(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.fish = make_fish(hparams)
        self.init_weights()
        self.result = []
    def forward(self,x):
        out = self.fish(x)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def loss_f(self, modely, targety):
        f = nn.CrossEntropyLoss()
        return f(modely, targety)

    def configure_optimizers(self):
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd, momentum=0.9)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=[30,40,50])
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "monitor": "val_loss"}}

    def step(self, x):
        img, label = x['img'], x['label'].view(-1)
        y_hat = self(img)
        loss = self.loss_f(y_hat, label)
        pred_c = y_hat.argmax(-1).detach().cpu().tolist()
        f1 = self.score_function(label.cpu().tolist(), pred_c)
        return loss, f1

    def score_function(self,pred, real):
        score = f1_score(real, pred, average="micro")
        return score

    def training_step(self, batch, batch_idx):
        loss,f1 = self.step(batch)
        self.log('trn_loss', loss, on_step=False, on_epoch=True)
        self.log('trn_f1',   f1, on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss,f1 = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_f1',   f1, on_step=False, on_epoch=True)
        return {'val_loss': loss,'f1':f1}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([op['val_loss'] for op in outputs]).mean()
        avg_f1 = torch.stack([op['f1'] for op in outputs]).mean()

        print("\n* EPOCH %s | val loss :{%4.4f} | val f1 :{%2.2f}" % (self.current_epoch, avg_loss, avg_f1))
        return {'loss': avg_loss}

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        pred_c = y_hat.argmax(-1).detach().cpu().tolist()
        self.result.extend(pred_c)

##


