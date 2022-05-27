python train.py --batch_size=256 --max_epochs=100 --lr=1e-2 --model=fishnet --num_c=64,128,256,512 --Ls_tail=3,4,6,2 --Ls_body=1,1,1 --Ls_head=1,1,1,1 --preact=True # fish99
# python train.py --batch_size=256 --max_epochs=100 --lr=1e-2 --model=fishnet --num_c=64,128,256,512 --Ls_tail=1,1,1,1 --Ls_body=1,1,1 --Ls_head=1,1,1,1 --preact=True # fish73
# python train.py --batch_size=256 --max_epochs=100 --lr=1e-2 --model=fishnet --num_c=64,128,256,512 --Ls_tail=3,4,6,3 --Ls_body=2,3,2 --Ls_head=3,4,6,3 --preact=True # fish150
