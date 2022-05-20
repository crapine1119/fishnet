import numpy as np
import pandas as pd
import cv2 as cv
import pickle
import random
import os
import matplotlib.pyplot as plt
from glob import glob as glob
from tqdm import tqdm as tqdm
#
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
#
import torch
from torch.utils.data import Dataset
from torchvision.utils import make_grid
##
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
    trnx = np.concatenate(trnx)
    trny = np.array(trny)

    # test set
    batch = unpickle(tst_file)
    tstx = batch[b'data']
    tsty = np.array(batch[b'labels'])
    return trnx,tstx,trny,tsty

class custom(Dataset):
    def __init__(self, imgs, labels, trans, train=True):
        super().__init__()
        self.imgs =imgs
        self.labels = labels
        self.trans = trans
        self.trans_test = A.Compose([A.Normalize(),ToTensorV2()])
        self.train=train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img = self.imgs[item].reshape(3,32,32)[::-1] # BGR to RGB
        img = np.transpose(img, (1,2,0))
        label = self.labels[item]
        mat = {}
        if self.train:
            if not self.trans is None:
                img = self.trans(image=img)['image']
            else:
                img = self.trans_test(image=img)['image']
            mat['label'] = torch.LongTensor(label)
        else:
            img = self.trans_test(image=img)['image']
        mat['img'] = torch.FloatTensor(img)
        return mat








