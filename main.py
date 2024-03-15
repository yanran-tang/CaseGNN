import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import json
import os
from tqdm import tqdm

from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.data.utils import load_graphs, save_graphs
from dgl.data import DGLDataset


from DATASET.data_load import SyntheticDataset, PoolDataset, collate
from model import CaseGNN, early_stopping

from train import forward

from torch.utils.tensorboard import SummaryWriter
import time
import logging

import argparse
parser = argparse.ArgumentParser()
## model parameters
parser.add_argument("--in_dim", type=int, default=768, help="Input_feature_dimension")
parser.add_argument("--h_dim", type=int, default=768, help="Hidden_feature_dimension")
parser.add_argument("--out_dim", type=int, default=768, help="Output_feature_dimension")
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout for embedding / GNN layer ")       
parser.add_argument("--num_head", default=1, type=int, help="Head number of GNN layer ")                            

## training parameters
parser.add_argument("--epoch", type=int, default=100, help="Training epochs")
parser.add_argument("--lr", type=float, default=1e-05, help="Learning rate")
parser.add_argument("--wd", default=1e-05, type=float, help="Weight decay if we apply some.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--temp", type=float, default=0.1, help="Temperature for relu")
parser.add_argument("--ran_neg_num", type=int, default=1, help="Random sampled case number")
parser.add_argument("--hard_neg", type=bool, default=True, help="Using bm25_neg or not")
parser.add_argument("--hard_neg_num", type=int, default=1, help="Bm25_neg case number")


## other parameters
parser.add_argument("--data", type=str, default='2022', help="coliee2022 or coliee2023")

args = parser.parse_args()

# Logger configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s')
logging.warning(args)

def main():

    log_dir = './Caselink_experiments/coliee'+args.data+'_bs'+str(args.batch_size)+'_dp'+str(args.dropout)+'_lr'+str(args.lr)+'_wd'+str(args.wd)+'_t'+str(args.temp)+'_headnum'+str(args.num_head)+'_hardneg'+str(args.hard_neg_num)+'_ranneg'+str(args.ran_neg_num)+'_'+time.strftime("%m%d-%H%M%S")
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## model initialization
    model = CaseGNN(args.in_dim, args.h_dim, args.out_dim, dropout=args.dropout, num_head=args.num_head)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    ## Dataset initialization
    
    # Train dataset
    train_dataset = SyntheticDataset("./Graph_generation/graph/graph_bin_"+args.data+"/bidirec_"+args.data+"train_fact_Synthetic.bin")
    train_graph = train_dataset.graph_list
    train_label = train_dataset.label_list
    train_sampler = SubsetRandomSampler(torch.arange(len(train_graph)))
    train_dataloader = GraphDataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=False, collate_fn=collate)

    train_sumfact_pool_dataset = PoolDataset("./Graph_generation/graph/graph_bin_"+args.data+"/bidirec_"+args.data+"train_fact.bin")
    train_referissue_pool_dataset = PoolDataset("./Graph_generation/graph/graph_bin_"+args.data+"/bidirec_"+args.data+"train_issue.bin")

    # Test dataset
    test_sumfact_dataset = SyntheticDataset("./Graph_generation/graph/graph_bin_"+args.data+"/bidirec_"+args.data+"test_fact_Synthetic.bin")

    test_sumfact_graph = test_sumfact_dataset.graph_list
    # test_sumfact_label = test_sumfact_dataset.label_list
    test_sumfact_sampler = SubsetRandomSampler(torch.arange(len(test_sumfact_graph)))
    
    ##Inference batch size
    if args.data == '2022':
        inference_bs = 1563
    elif args.data == '2023':
        inference_bs = 1335
    test_dataloader = GraphDataLoader(
        test_sumfact_dataset, sampler=test_sumfact_sampler, batch_size=inference_bs, drop_last=False, collate_fn=collate, shuffle=False)

    test_sumfact_pool_dataset = PoolDataset("./Graph_generation/graph/graph_bin_"+args.data+"/bidirec_"+args.data+"test_fact.bin")
    test_referissue_pool_dataset = PoolDataset("./Graph_generation/graph/graph_bin_"+args.data+"/bidirec_"+args.data+"test_fact.bin")

    ## load train label
    train_labels = {}
    with open('./label/task1_train_labels_'+args.data+'.json', 'r')as f:
        train_labels = json.load(f)
        f.close() 

    bm25_hard_neg_dict = {}
    with open('./label/hard_neg_top50_train_'+args.data+'.json', 'r')as file:
        for line in file.readlines():
            dic = json.loads(line)
            bm25_hard_neg_dict.update(dic)
        file.close() 

    # ## load test label
    test_labels = {}
    with open('./label/task1_test_labels_'+args.data+'.json', 'r')as f:
        test_labels = json.load(f)
        f.close()    

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.warning('logging to {}'.format(log_dir))

    for epoch in tqdm(range(args.epoch)):
        print('Epoch:', epoch)
        forward(model, device, writer, train_dataloader, train_sumfact_pool_dataset, train_referissue_pool_dataset, train_labels, epoch, args.temp, bm25_hard_neg_dict, args.hard_neg, args.hard_neg_num, train_flag=True, optimizer=optimizer)
        with torch.no_grad():            
            forward(model, device, writer, test_dataloader, test_sumfact_pool_dataset, test_referissue_pool_dataset, test_labels, epoch, args.temp, bm25_hard_neg_dict, args.hard_neg, args.hard_neg_num, train_flag=False, optimizer=optimizer)

if __name__ == '__main__':
    main()