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

import requests 
import re

import numpy as np

from google_images_download import google_images_download



client = Groq(api_key = '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')


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
        # allready done
        pass

with open('nodeID_to_test.pkl', 'rb') as f:
    text_nodeID = pickle.load(f)

print('loaded nodes')

with open('edgeID_to_test.pkl', 'rb') as f:
    text_edgeID = pickle.load(f)

print('loaded edges')

root = './data/Wikidata5M'

# Load the dataset
dataset = Wikidata5M(root)

print(len(dataset))

data = dataset[0]  # The dataset is usually accessed as a list of graphs

print(data)
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Edge index: {data.edge_index.shape}')
print(f'Node features: {data.x.shape}')



source_nodes = data.edge_index[0]
non_repeat_source_nodes = torch.unique(source_nodes)


UserPrompt = []
prompt_with_neigh = []
prompt_no_neigh = []
LLAMA_neigh = []
LLAMA_no_neigh = []
LLAMA_alone = []


# indices = [11,
# 18,
# 23,
# 25,
# 29,
# 37,
# 39,
# 47,
# 49,
# 53,
# 55,
# 59,
# 60,
# 65,
# 68,
# 74,
# 76]



def download_image(prompt, name, folder, model='flux-pro'):
    url = f"https://pollinations.ai/p/{prompt}&model={model}"
    response = requests.get(url)
    with open(osp.join(folder, name), 'wb') as file:
        file.write(response.content)
    print('Image downloaded!')


for k in range(230,260):#indices: #non_repeat_source_nodes[0:100]:
    print(k)

    main_text = text_nodeID[k]#.item()]

    short_main_text = ' '.join(main_text.split(' ')[:6])
    print(short_main_text)



    chat_completion_alone = client.chat.completions.create(messages = [{
                "role": "user", 
                "content": f"Provide a detailed prompt for realistic image generation of {re.sub(r'[^\w\s]','',short_main_text)}."
                }], model = 'llama3-70b-8192')

    LLAMAprompt = chat_completion_alone.choices[0].message.content
    # download_image(LLAMAprompt, str(k) + '_' + re.sub(r'[^\w\s]','',short_main_text) + '.jpg', '2gen_img_LLAMAuserprompt')
    # download_image(re.sub(r'[^\w\s]','',short_main_text), str(k) + '_' + re.sub(r'[^\w\s]','',short_main_text) + '.jpg', '2gen_img_userprompt')

    
    LLAMA_alone.append(LLAMAprompt)
    UserPrompt.append(re.sub(r'[^\w\s]','',short_main_text))


    neighbors = data.edge_index[1, data.edge_index[0] == k]
    edge_type = data.edge_type[data.edge_index[0] == k]

    n_text = ''
    for n in range(len(edge_type)):
        if n >10:
             break
        nitem = neighbors[n].item()
        nt = ' '.join(text_nodeID[nitem].split(' ')[:6])
        e = edge_type[n].item()      
        n_text += f" {re.sub(r'[^\w\s]','',short_main_text)} is {text_edgeID[e+1]} {nt}."
        

    chat_completion = client.chat.completions.create(messages = [{
                "role": "user", 
                "content": (f"Provide a detailed prompt for realistic image generation of {re.sub(r'[^\w\s]','',short_main_text)}."
                            f"Use information about appearance and environment "
                            f"from this data: {main_text} {n_text}"
                            )
                }], model = 'llama3-70b-8192')

    LLAMAprompt = chat_completion.choices[0].message.content
    # download_image(LLAMAprompt, str(k) + '_' + re.sub(r'[^\w\s]','',short_main_text) + '.jpg', '2gen_img_LLAMAgraph')
    # download_image(f'{main_text} {n_text}', str(k) + '_' + re.sub(r'[^\w\s]','',short_main_text) + '.jpg', '2gen_img_graph')
    # download_image(f'{main_text}', str(k) + '_' + re.sub(r'[^\w\s]','',short_main_text) + '.jpg', '2gen_img_graphnoneigh')

    
    LLAMA_neigh.append(LLAMAprompt)

    prompt_with_neigh.append(f'{main_text} {n_text}')
    prompt_no_neigh.append(f'{main_text}')

    chat_completion = client.chat.completions.create(messages = [{
                "role": "user", 
                "content": (f"Provide a detailed prompt for realistic image generation of {re.sub(r'[^\w\s]','',short_main_text)}."
                            f"Use information about appearance and environment "
                            f"from this data: {main_text}"
                            )
                }], model = 'llama3-70b-8192')

    LLAMAprompt = chat_completion.choices[0].message.content
    # download_image(LLAMAprompt, str(k) + '_' + re.sub(r'[^\w\s]','',short_main_text) + '.jpg', '2gen_img_LLAMAgraphnoneigh')

    LLAMA_no_neigh.append( LLAMAprompt)

df = pd.DataFrame(np.array([j for j in range(230, 260)]))

df['UserPrompt'] = UserPrompt
df['DetailsNodeNeighbors'] = prompt_with_neigh
df['DetailsNode'] = prompt_no_neigh
df['LLAMANodeNeighbors'] = LLAMA_neigh
df['LLAMANode'] = LLAMA_no_neigh
df['LLAMAUserPrompt'] = LLAMA_alone

df.to_csv('230-260_withprompts.csv', index=False)