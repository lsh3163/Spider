import pickle
import gzip

import os
import os.path as osp
import torch_geometric
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Optional

import networkx as nx

import torch

import torch_geometric.transforms as T
import torch_geometric.utils as utils

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
)

import collections

from torch_geometric.utils import k_hop_subgraph

from groq import Groq

# from transformers import BertTokenizer,BertModel

# from sentence_transformers import SentenceTransformer
import pandas as pd
import time

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
# model = BertModel.from_pretrained("bert-base-uncased")

# sentence = 'I really enjoyed this movie a lot.'

# tokens = tokenizer.tokenize(sentence)
# tokens = ['[CLS]'] + tokens + ['[SEP]']
# T=500
# padded_tokens = tokens + ['[PAD]' for _ in range(T-len(tokens))]
# attn_mask = [ 1 if token != '[PAD]' else 0 for token in padded_tokens  ]

# seg_ids = [0 for _ in range(len(padded_tokens))]

# sent_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
# token_ids = torch.tensor(sent_ids).unsqueeze(0) 
# attn_mask = torch.tensor(attn_mask).unsqueeze(0) 
# seg_ids   = torch.tensor(seg_ids).unsqueeze(0)

# output = model(token_ids, attention_mask=attn_mask,token_type_ids=seg_ids)
# last_hidden_state, pooler_output = output[0], output[1]

# print(last_hidden_state.shape) 
# print(pooler_output.shape) 

# exit()


client = Groq(api_key = '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')





## run first time for processing
class Wikidata5M(InMemoryDataset):

    def __init__(
        self,
        root: str,
        setting: str = 'transductive',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        if setting not in {'transductive', 'inductive'}:
            raise ValueError(f"Invalid 'setting' argument (got '{setting}')")

        self.setting = setting

        self.urls = [
            ('https://www.dropbox.com/s/7jp4ib8zo3i6m10/'
             'wikidata5m_text.txt.gz?dl=1'),
            'https://uni-bielefeld.sciebo.de/s/yuBKzBxsEc9j3hy/download',
        ]
        if self.setting == 'inductive':
            self.urls.append('https://www.dropbox.com/s/csed3cgal3m7rzo/'
                             'wikidata5m_inductive.tar.gz?dl=1')
        else:
            self.urls.append('https://www.dropbox.com/s/6sbhm0rwo4l73jq/'
                             'wikidata5m_transductive.tar.gz?dl=1')

        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'wikidata5m_text.txt.gz',
            'download',
            f'wikidata5m_{self.setting}_train.txt',
            f'wikidata5m_{self.setting}_valid.txt',
            f'wikidata5m_{self.setting}_test.txt',
        ]

    @property
    def processed_file_names(self) -> str:
        return f'{self.setting}_data.pt'

    def download(self) -> None:
        for url in self.urls:
            download_url(url, self.raw_dir)
        path = osp.join(self.raw_dir, f'wikidata5m_{self.setting}.tar.gz')
        extract_tar(path, self.raw_dir)
        os.remove(path)

    def process(self) -> None:
        import gzip

        entity_to_id: Dict[str, int] = {}
        with gzip.open(self.raw_paths[0], 'rt', encoding='utf-8') as f: ##  ENCODING ADDED, MOD FROM PYTORCH GEOMATRIC
            for i, line in enumerate(f):
                values = line.strip().split('\t')
                entity_to_id[values[0]] = i

        x = torch.load(self.raw_paths[1])

        edge_indices = []
        edge_types = []
        split_indices = []

        rel_to_id: Dict[str, int] = {}
        for split, path in enumerate(self.raw_paths[2:]): 
            with open(path, 'r', encoding = "iso-8859-1") as f: ## ENCODING ADDED, MOD FROM PYTORCH GEOMATRIC
                for line in f:
                    head, rel, tail = line[:-1].split('\t')
                    src = entity_to_id[head]
                    dst = entity_to_id[tail]
                    edge_indices.append([src, dst])
                    if rel not in rel_to_id:
                        rel_to_id[rel] = len(rel_to_id)
                    edge_types.append(rel_to_id[rel])
                    split_indices.append(split)
                    
                
        edge_index = torch.tensor(edge_indices).t().contiguous()
        edge_type = torch.tensor(edge_types)
        split_index = torch.tensor(split_indices)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            train_mask=split_index == 0,
            val_mask=split_index == 1,
            test_mask=split_index == 2,
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])







## check nodes based on unlabeled edges
root = './data/Wikidata5M'

node_path = osp.join(root, 'raw', 'wikidata5m_text.txt.gz')

with gzip.open(node_path, 'rt', encoding='utf-8') as f:
    for i, line in enumerate(f):
        values = line.strip().split('\t')
        if values[0] in {'Q5230422', 'Q1568', 'Q5348986', 'Q704525','Q4917', 'Q11110','Q2674262', 'Q28133204'}:
            print(values[0], values[1][:100])

# exit()



# check other data for edge labels
# root = './data/Wikidata5M'

# node_path = osp.join(root, 'raw', 'wikidata5m_text_separate_dwnld.txt.gz')
# # node_path = osp.join(root, 'raw', 'wikidata5m_text.txt.gz')

# with gzip.open(node_path, 'rt', encoding='utf-8') as f:
#     for i, line in enumerate(f):
#         values = line.strip().split('\t')
#         if values[0][0] != 'Q': 
#             print(values)

# exit()


## creating label files

root = './data/Wikidata5M'

text_nodeID: Dict[int, str] = {}

node_path = osp.join(root, 'raw', 'wikidata5m_text.txt.gz')

with gzip.open(node_path, 'rt', encoding='utf-8') as f:
    for i, line in enumerate(f):
        values = line.strip().split('\t')
        text_nodeID[i] = values[1]

print('saving')

# Save the dictionary to a file
with open('nodeID_to_test.pkl', 'wb') as f:
    pickle.dump(text_nodeID, f)

# exit()



## creating edge labels 

root = './data/Wikidata5M'
alias_path = osp.join(root, 'raw', 'wikidata5m_alias.tar.gz')
text_edge: Dict[str, str] = {}

with gzip.open(alias_path, 'rt', encoding = "iso-8859-1")  as f: #iso-8859-1
    for i, line in enumerate(f):
        
        if line[0] == 'P': 
          values = line.strip().split('\t')
          text_edge[values[0]] = values[1]
     

text_edge.update({'P2439': 'language', 
 'P1962': 'under the patronage of',
  'P489': 'represented by',
  'P3484': 'variant of or similar to'
  }) # manually created since missing

# print(text_edge)

text_edgeID: Dict[int, str] = {}

rel_to_id: Dict[str, int] = {}


raw_paths = [
    f'wikidata5m_transductive_train.txt',
    f'wikidata5m_transductive_valid.txt',
    f'wikidata5m_transductive_test.txt',
]


for split, file in enumerate(raw_paths):
    path = osp.join(root, 'raw', file)
    with open(path, 'r') as f:
       
        for line in f:

            head, rel, tail = line[:-1].split('\t')

            if rel not in rel_to_id:
                rel_to_id[rel] = len(rel_to_id)
                try:
                    text_edgeID[len(rel_to_id)] = text_edge[rel]
                except: 
                    print(head, rel, tail)
print(len(rel_to_id))
print(len(text_edge))
print(len(text_edgeID))


print('saving')

# Save the dictionary to a file
with open('edgeID_to_test.pkl', 'wb') as f:
    pickle.dump(text_edgeID, f)

# exit()




# root = './data/Wikidata5M'

# # Load the dataset
# dataset = Wikidata5M(root)

# print(len(dataset))

# data = dataset[0]  # The dataset is usually accessed as a list of graphs

# print(data)
# print(f'Number of nodes: {data.num_nodes}')
# print(f'Number of edges: {data.num_edges}')
# print(f'Edge index: {data.edge_index.shape}')
# print(f'Node features: {data.x.shape}')
