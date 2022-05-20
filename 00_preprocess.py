from sklearn.model_selection import train_test_split
import argparse
from utils import *
##
parser = argparse.ArgumentParser(description='configs')
# configs for preprocess
parser.add_argument('--resize',         default=32,         type=int,      help = '')
# hparams = parser.parse_args()
hparams = parser.parse_args(args=[]) # 테스트용

## check image
seed_everything()

rdir = r'D:\cv\Dataset\cifar-10-python\cifar-10-batches-py'
names = 'airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck'.split(',')
n2l = {n:e for e,n in enumerate(names)}
l2n = {e:n for e,n in enumerate(names)}

# check_img(rdir)
##

trans = A.Compose([A.Normalize(),
                   ToTensorV2()])

trnx,tstx,trny,tsty = load_cifar(rdir)

trnx,valx,trny,valy = train_test_split(trnx,trny,test_size=0.2)

trn_set = custom(trnx,trny,trans=trans,train=True)
val_set = custom(trnx,trny,trans=trans,train=True)
tst_set = custom(trnx,trny,trans=trans,train=False)