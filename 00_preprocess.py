from sklearn.model_selection import train_test_split
import argparse
from utils import *
##
parser = argparse.ArgumentParser(description='configs')
# configs for preprocess
parser.add_argument('--resize',         default=32,         type=int,      help = '')
parser.add_argument('--in_c',           default=3,          type=int,      help = '')
parser.add_argument('--out_c',          default=10,         type=int,      help = '')
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
trans = A.Compose([A.Resize(96,96, interpolation=cv.INTER_AREA),
                   A.Normalize(mean=[0.4914, 0.4822, 0.4465],
                               std=[0.2470, 0.2435, 0.2616]),
                   ToTensorV2()])

trnx,tstx,trny,tsty = load_cifar(rdir)

# torch.FloatTensor((trnx/255).reshape(-1,3,32,32)).mean(dim=[0,-1,-2])
# torch.FloatTensor((trnx/255).reshape(-1,3,32,32)).std(dim=[0,-1,-2])


trnx,valx,trny,valy = train_test_split(trnx,trny,test_size=0.2)

trn_set = custom(trnx,trny,trans=trans,train=True)
val_set = custom(trnx,trny,trans=trans,train=True)
tst_set = custom(trnx,trny,trans=trans,train=False)
