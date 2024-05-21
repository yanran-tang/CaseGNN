## To use augmentation on different samples, the last argument can be (--pos_aug), (--ran_aug) or（--pos_aug --ran_aug）

python main_casegnn2plus.py --in_dim=768 --h_dim=768 --out_dim=768 --dropout=0.1 --num_head=1 --epoch=1000 --lr=1e-4 --wd=1e-4 --batch_size=32 --temp=0.1 --ran_neg_num=1 --hard_neg_num=0 --aug_edgedrop=0.1 --aug_featmask_node=0 --aug_featmask_edge=0 --data=2022 --ran_aug