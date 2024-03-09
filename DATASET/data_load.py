import dgl
from dgl.data.utils import load_graphs
from dgl.data import DGLDataset

import json
import torch

import os
from tqdm import tqdm


class PoolDataset(DGLDataset):
    def __init__(self, file_path):
        self.graph_and_label = load_graphs(file_path)
        super(PoolDataset, self).__init__(name='Pool')

    def process(self):
        case_pool, pool_label = self.graph_and_label
        CaseGraph = {}
        graphs = case_pool
        labels = pool_label['name_list'].tolist()
        for y in range(len(labels)):
            CaseGraph.update({str(int(labels[y])).zfill(6): graphs[y]})
        self.graphs = CaseGraph

        label_dict = {}
        labels = pool_label['name_list'].tolist()
        for x in range(len(labels)):
            label_dict.update({str(int(labels[x])).zfill(6): [str(int(labels[x])).zfill(6)]})
        self.labels = label_dict

        self.graph_list = case_pool
        self.label_list = pool_label['name_list'].tolist()
        # self.label_list = torch.LongTensor(self.label_list)

    def __getitem__(self, i):
        return self.graph_list[i], self.label_list[i]

    def __len__(self):
        return len(self.graphs)


class SyntheticDataset(DGLDataset):
    def __init__(self, file_path):
        self.graph_and_label = load_graphs(file_path)
        super(SyntheticDataset, self).__init__(name='Synthetic')

    def process(self):
        CaseGraph = {}
        graphs = self.graph_and_label[0]
        labels = self.graph_and_label[1]['glabel'].tolist()
        for y in range(len(labels)):
            CaseGraph.update({str(int(labels[y])).zfill(6): graphs[y]})
        self.graphs = CaseGraph

        label_dict = {}
        labels = self.graph_and_label[1]['glabel'].tolist()
        for x in range(len(labels)):
            label_dict.update({str(int(labels[x])).zfill(6): [str(int(labels[x])).zfill(6)]})
        self.labels = label_dict

        self.graph_list = self.graph_and_label[0]
        self.label_list = self.graph_and_label[1]['glabel'].tolist()
        # self.label_list = torch.LongTensor(self.label_list)

    def __getitem__(self, i):
        return self.graph_list[i], self.label_list[i]

    def __len__(self):
        return len(self.graphs)


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


def load_data(key, tokenizer, max_length=None, construct_textgraph=False, n_jobs=1,
              force_lowercase=False, raw=False):
    raw_documents = []
    data_path = './data/no_fr_txt_221222'
    files = os.listdir(data_path)
    for pfile in tqdm(files[:]):
        with open(os.path.join(data_path, pfile), 'r') as f:
            jsfile = f.read()
            f.close()
        raw_documents.append(jsfile)

    N = len(raw_documents)

    labels = []
    # train_mask, test_mask = torch.zeros(N, dtype=torch.bool), torch.zeros(N, dtype=torch.bool)

    docs = [tokenizer.encode(raw_doc) for raw_doc in raw_documents]

    print("Encoding labels...")
    label2index = {label: idx for idx, label in enumerate(set(labels))}
    label_ids = [label2index[label] for label in tqdm(labels)]

    return docs, label_ids, label2index