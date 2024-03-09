import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel
import torch.nn as nn

import torch
import dgl
import os
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info, save_graphs, load_graphs
from tqdm import tqdm
import json

import dgl.data
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='2022', help="coliee2022 or coliee2023")
parser.add_argument("--dataset", type=str, default='train', help="train or test")
parser.add_argument("--feature", type=str, default='fact', help="fact or issue")
args = parser.parse_args()


# Load Model
device = torch.device('cuda')
model_name = 'CSHaitao/SAILER_en_finetune'
model = AutoModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


candidate_matrix = torch.load("./PromptCase/promptcase_embedding/"+args.data+"_"+args.dataset+"_fact_issue_cross_embedding.pt")

with open("./PromptCase/promptcase_embedding/"+args.data+"_"+args.dataset+"_fact_issue_cross_embedding_case_list.json", "rb")as fIn:
    candidate_matrix_index = json.load(fIn) 

if args.feature == 'fact':
    ie_path = "./Information_extraction/coliee"+args.data+"_ie/coliee"+args.data+args.dataset+"_sum/result/"
    embedding_index = 0
elif args.feature == 'issue':
    ie_path = "./Information_extraction/coliee"+args.data+"_ie/coliee"+args.data+args.dataset+"_refer/result/"
    embedding_index = 1

file_list = os.listdir(ie_path)

graph_num = 0
graph_list = []
graph_labels = {}
graph_name_list = []
zero_file = []

model.eval()
with torch.no_grad():
    for file in tqdm(file_list):
        graph_num += 1
        file_name = file.split('_')[-1].split('.')[0]
        ## case_embedding_format = [fact_embedding, issue_embedding, cross_embedding]
        promptcase_embedding = candidate_matrix[0][embedding_index][candidate_matrix_index.index(file_name+'.txt')]
        graph_name_list.append(int(file_name))
        list_node1 = []
        list_node2 = []
        list_relation = []
        index_dict = {}
        node_num = -1
        Relation_embedding_weights = []
        node_embedding_weights = []
        split_txt_list = []
                
        with open(ie_path+file, "r") as f:
            relation_triplets = f.readlines()
            for line in relation_triplets:
                if line == 'Type,Entity1,Relationship,Type,Entity2\n':
                    node_num += 1
                    index_dict.update({'promptcase_node': node_num})
                    node_embedding_weights.append(promptcase_embedding)
                    list_node1.append(node_num)
                    list_node2.append(node_num)
                    Relation_embedding_weights.append(promptcase_embedding)
                    list_node2.append(node_num)
                    list_node1.append(node_num)
                    Relation_embedding_weights.append(promptcase_embedding)
                    
                else:
                    if '×' in line:
                        line = line.replace('×','')
                    a_1 = line[:-1].split(',')
                    split_txt = [a_1[1], a_1[2], a_1[4]]
                    if split_txt in split_txt_list:
                        continue
                    else:
                        tokenized = tokenizer(split_txt, return_tensors='pt', padding=True, truncation=True).to(device)
                        embedding = model(**tokenized)
                        cls_embedding = embedding[0][:,0] ##cls token embedding [1,768]
                        cls_embedding = cls_embedding.to('cpu')
                        Entity_1_embedding = cls_embedding[0]
                        Entity_2_embedding = cls_embedding[2]
                        Relation = cls_embedding[1]
                        if a_1[1] in index_dict.keys():
                            Entity_1 = index_dict[a_1[1]]
                        else:
                            node_num += 1
                            Entity_1 = node_num
                            index_dict.update({a_1[1]: Entity_1})
                            node_embedding_weights.append(Entity_1_embedding)
                        if a_1[4] in index_dict.keys():
                            Entity_2 = index_dict[a_1[4]]
                        else:
                            node_num += 1
                            Entity_2 = node_num
                            index_dict.update({a_1[4]: Entity_2})
                            node_embedding_weights.append(Entity_1_embedding)
                        list_node1.append(Entity_1)
                        list_node2.append(Entity_2)
                        Relation_embedding_weights.append(Relation)
                        
                        list_node1.append(Entity_2)
                        list_node2.append(Entity_1)
                        Relation_embedding_weights.append(Relation)
                        
                        list_node1.append(index_dict['promptcase_node'])
                        list_node2.append(Entity_1)
                        Relation_embedding_weights.append(Entity_1_embedding)
                        
                        list_node1.append(Entity_1)
                        list_node2.append(index_dict['promptcase_node'])
                        Relation_embedding_weights.append(Entity_1_embedding)
                        
                        list_node1.append(index_dict['promptcase_node'])
                        list_node2.append(Entity_2)
                        Relation_embedding_weights.append(Entity_2_embedding)
                        
                        list_node1.append(Entity_2)
                        list_node2.append(index_dict['promptcase_node'])
                        Relation_embedding_weights.append(Entity_2_embedding)
                        split_txt_list.append(split_txt)

            f.close()
            print(file)

        # Graph Construction
        g = dgl.graph((list_node1, list_node2))
        g_1 = g

        if len(Relation_embedding_weights) == 0:
            b = 0
            c = 0
            print(file_name, ': zero node and edge')
            zero_file.append(file_name)
        else:
            print('Relation num:', len(Relation_embedding_weights))
            print('Node num:', len(node_embedding_weights))
            b = torch.stack(Relation_embedding_weights)
            c = torch.stack(node_embedding_weights)

            g_1.ndata['w'] = c
            g_1.edata['w'] = b
            graph_list.append(g_1)

tensor_graph_name = torch.FloatTensor(graph_name_list)
graph_labels.update({'name_list': tensor_graph_name})

save_graphs("./Graph_generation/graph/graph_bin_"+args.data+"/bidirec_"+args.data+args.dataset+"_"+args.feature+".bin", graph_list, graph_labels)
print('Graph saved.')
if len(zero_file) != 0:
    print(zero_file)

# test/train synthetic graph construction
if args.dataset == 'test':
    graph_label = {"glabel": tensor_graph_name}
    save_graphs("./Graph_generation/graph/graph_bin_"+args.data+"/bidirec_"+args.data+args.dataset+"_"+args.feature+"_Synthetic.bin", graph_list, graph_label)
elif args.dataset == 'train':
    CaseGraph = {}
    labels = tensor_graph_name.tolist()
    for i in range(len(labels)):
        CaseGraph[str(int(labels[i])).zfill(6)] = graph_list[i]
    with open('./label/task1_train_labels_'+args.data+'.json', 'r') as f:
        noticed_case_list = json.load(f)
        f.close()
    query_graph_dict = {}
    query_graph_list = []
    query_graph_label = []
    for key, value in noticed_case_list.items():
        k = key.split('.')[0]
        query_graph_dict.update({k: (CaseGraph[str(k)])})
        query_graph_list.append(CaseGraph[str(k)])
        query_graph_label.append(int(k))
    graph_labels = {"glabel": torch.Tensor(query_graph_label)}

    save_graphs("./Graph_generation/graph/graph_bin_"+args.data+"/bidirec_"+args.data+args.dataset+"_"+args.feature+"_Synthetic.bin", query_graph_list, graph_labels)

