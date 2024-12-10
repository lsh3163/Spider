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


## checking edges w/o labels

# Q5230422 P2439 Q1568
# Q5348986 P1962 Q704525
# Q4917 P489 Q11110
# Q2674262 P3484 Q28133204


# root = './data/Wikidata5M'

# node_path = osp.join(root, 'raw', 'wikidata5m_text.txt.gz')

# with gzip.open(node_path, 'rt', encoding='utf-8') as f:
#     for i, line in enumerate(f):
#         values = line.strip().split('\t')
#         if values[0] in {'Q5230422', 'Q1568', 'Q5348986', 'Q704525','Q4917', 'Q11110','Q2674262', 'Q28133204'}:
#             print(values[0], values[1][:100])

# exit()



## check other data for edge labels
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

# root = './data/Wikidata5M'

# text_nodeID: Dict[int, str] = {}

# node_path = osp.join(root, 'raw', 'wikidata5m_text.txt.gz')

# with gzip.open(node_path, 'rt', encoding='utf-8') as f:
#     for i, line in enumerate(f):
#         values = line.strip().split('\t')
#         text_nodeID[i] = values[1]

# print('saving')

# # Save the dictionary to a file
# with open('nodeID_to_test.pkl', 'wb') as f:
#     pickle.dump(text_nodeID, f)

# exit()



## creating edge labels 

# root = './data/Wikidata5M'
# alias_path = osp.join(root, 'raw', 'wikidata5m_alias.tar.gz')
# text_edge: Dict[str, str] = {}

# with gzip.open(alias_path, 'rt', encoding = "iso-8859-1")  as f: #iso-8859-1
#     for i, line in enumerate(f):
        
#         if line[0] == 'P': 
#           values = line.strip().split('\t')
#           text_edge[values[0]] = values[1]
     

# text_edge.update({'P2439': 'language', 
#  'P1962': 'under the patronage of',
#   'P489': 'represented by',
#   'P3484': 'variant of or similar to'
#   }) # manually created since missing

# # print(text_edge)

# text_edgeID: Dict[int, str] = {}



# rel_to_id: Dict[str, int] = {}


# raw_paths = [
#     f'wikidata5m_transductive_train.txt',
#     f'wikidata5m_transductive_valid.txt',
#     f'wikidata5m_transductive_test.txt',
# ]


# for split, file in enumerate(raw_paths):
#     path = osp.join(root, 'raw', file)
#     with open(path, 'r') as f:
       
#         for line in f:

#             head, rel, tail = line[:-1].split('\t')

#             if rel not in rel_to_id:
#                 rel_to_id[rel] = len(rel_to_id)
#                 try:
#                     text_edgeID[len(rel_to_id)] = text_edge[rel]
#                 except: 
#                     print(head, rel, tail)
# print(len(rel_to_id))
# print(len(text_edge))
# print(len(text_edgeID))


# print('saving')

# # Save the dictionary to a file
# with open('edgeID_to_test.pkl', 'wb') as f:
#     pickle.dump(text_edgeID, f)

# exit()




with open('nodeID_to_test.pkl', 'rb') as f:
    text_nodeID = pickle.load(f)

# print(text_nodeID[10])
# exit()


# for k, v in text_nodeID.items():
#     if 'The Dassault Falcon 900, commonly' in v: 
#         print(k)

# # print(text_nodeID[ii])
# exit()


# exit()
print('loaded nodes')

with open('edgeID_to_test.pkl', 'rb') as f:
    text_edgeID = pickle.load(f)

print('loaded edges')

# text_nodeID 
# text_edgeID


# from torch_geometric.datasets import Wikidata5M

# Specify the directory where the dataset should be downloaded
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

# print(data.x[0])
# print(data.edge_type[0:10])
# print(data.edge_index[:,0:10])
# print(f'Labels: {data.y.shape}')


# import networkx as nx
# import demon as d
# import igraph as ig
# import leidenalg
# import random 

# g = utils.to_networkx(data)
# print(g.number_of_nodes())

# num_nodes_to_pick = 5
# random_nodes = random.sample(g.nodes(), num_nodes_to_pick)

# total_subgraph_nodes = []
# total_communities = []
# total_duplicates = []
# total_correct = []

# for node in random_nodes: 

#     eg = nx.ego_graph(g, 10, radius = 3, center=True, undirected = True)
#     print('total nodes subgraph: ', eg.number_of_nodes())
#     total_subgraph_nodes.append(eg.number_of_nodes())

#     dm = d.Demon(graph=eg, epsilon=0.25, min_community_size=3) #, file_output=True)
#     coms = dm.execute()
#     print('number communities: ', len(coms))
#     total_communities.append(len(coms))

#     result = {}

#     # Iterate through the list of tuples
#     for index, tpl in enumerate(coms):
#         for num in tpl:
#             if num in result: 
#                 result[num] = result[num] + [index]
#             else: 
#                 result[num] = [index]

#     duplicates = 0
#     for val in result.values():
#         if len(val) > 1: 
#             duplicates += 1
#     print('duplicates DEMON: ', duplicates)

#     total_duplicates.append(duplicates)
    

#     avg_embed = torch.zeros(len(coms), 384)
#     for c in range(len(coms)): 
#         for other_node in coms[c]: # pick that community
#             avg_embed[c,:] = avg_embed[c,:] + data.x[other_node]
#         avg_embed[c,:] = avg_embed[c,:]/len(coms[c])


#     correct = 0
#     egs = random.sample(eg.nodes(), 1000)
#     for ee in egs: 
#         compare_to = data.x[ee]
#         value, k_ = torch.max(torch.sum(compare_to*avg_embed, dim=-1), 0)
#         if k_ in result[ee]:
#             correct += 1

#     print('percent matched: ', correct/1000)
#     total_correct.append(correct/1000)



# print('--------------------')
# print(total_subgraph_nodes)
# print(total_communities)
# print(total_duplicates)
# print(total_correct)

# exit()

# # g = ig.Graph(edges=data.edge_index.T.tolist(), directed=False)
# h = ig.Graph.from_networkx(eg)


# print('graph made')

# partition = leidenalg.find_partition(
#     h, 
#     leidenalg.ModularityVertexPartition,  # Maximizes modularity by default
# )

# partition = torch.tensor(partition.membership)

# binst = torch.bincount(partition.to(torch.int32))
# # print(binst)
# print(len(binst))


# # ig.plot(partition) 
# # plt.axis("off")
# # plt.show()

# # torch.tensor(partition)

# # print(len(partition))
# exit()

















### TODO: Compare different access approaches: speed vs accuracy

# ## cluster by fixed node # 
# ## cluster by fixed distance


# import networkx as nx
# import demon as d

# g = utils.to_networkx(data)
# dm = d.Demon(graph=g, epsilon=0.25, min_community_size=3, file_output=True)
# coms = dm.execute()
# print('number communities: ', len(coms))


# import igraph as ig

# # Create igraph Graph from edge list

# g = ig.Graph(edges=data.edge_index.T.tolist(), directed=False)


# import leidenalg

# print('graph made')

# partition = leidenalg.find_partition(
#     g, 
#     leidenalg.ModularityVertexPartition,  # Maximizes modularity by default
# )

# partition = torch.tensor(partition.membership)

# binst = torch.bincount(partition.to(torch.int32))
# print(binst)
# print(len(binst))

# # Extract community assignments
# # communities = partition.membership
# # print("Community assignments:", communities)
# exit()


## plotting node imbalance (to fix plot)
# edge_index = data.edge_index 

# degrees = utils.degree(edge_index[0])
# print(torch.bincount(degrees.to(torch.int32)))
# indices = torch.nonzero(torch.bincount(degrees.to(torch.int32)))

# print(indices)
# exit()
# plt.hist(degrees.numpy(), bins=20)  # Adjust the number of bins as needed
# plt.xlabel('Degree')
# plt.ylabel('Frequency')
# plt.title('Distribution of Edges per Node')
# plt.show()


# Baseline: since care about node with best feature = kmeans clustering
## to fix: choose subgraph for true comparison

# import faiss
# kmeans = faiss.Kmeans(d=384, k=2000, niter=2000, verbose=True)
# kmeans.train(data.x.numpy().astype('float32'))

# # Get cluster centroids
# centroids = kmeans.centroids
# print(centroids)
# exit()

############
# Kmeans always same

# from torch_kmeans import KMeans

# model = KMeans(n_clusters=4).cuda()

# torch.reshape(data.x, (857, 5619, 384))

# torch.manual_seed(0)
# result = model((data.x[:10000,:]).cuda().unsqueeze(0))
# # print(result.labels.shape)
# # print(result.labels[:5])
# print(torch.bincount(result.labels[0, :]))

# torch.manual_seed(1)
# result = model((data.x[:10000,:]).cuda().unsqueeze(0))
# # print(result.labels.shape)
# # print(result.labels[:5])
# print(torch.bincount(result.labels[0, :]))

# torch.manual_seed(2)
# result = model((data.x[:10000,:]).cuda().unsqueeze(0))
# # print(result.labels.shape)
# # print(result.labels[:5])
# print(torch.bincount(result.labels[0, :]))

# ### batch don't improve OOM
# # result = model(torch.reshape(data.x, (857, 5619, 384)).cuda())
# # # print(result.labels.shape)
# # # print(result.labels[:5])
# # print(torch.bincount(result[0]))

# exit()
####################











# sentences = ['Cape Glossy Starling', 'Shrike', 'Bulbul']

# model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-12-v3')
# embeddings = torch.tensor(model.encode(sentences))
# print(embeddings.shape)


# sim_score = []
# prompt_with_neigh = []
# prompt_no_neigh = []
# LLAMA_neigh = []
# LLAMA_no_neigh = []
# LLAMA_alone = []


# for ee in range(len(sentences)): 
#     e = embeddings[ee] 
#     print(sentences[ee])

#     start_time = time.time() # check time to get max
#     value, k = torch.max(torch.sum(e*data.x, dim=-1), 0)
#     print(time.time()-start_time)
#     print(value)
#     sim_score.append(value.item())










## actually useful: Cape Glossy Starling, Shrike, Bulbul

### Ground truth:
### Comparing prompt from our list to all other nodes to extract most similar by embedding
### Getting individual node and neighbors (full text node + 6 words). Generating prompt based on that + saving the text

# df = pd.read_csv('prompts.csv')

# listslists = df.values.tolist()
# sentences = [l[0] for l in listslists]

# model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-12-v3')
# embeddings = torch.tensor(model.encode(sentences))
# print(embeddings.shape)


# sim_score = []
# prompt_with_neigh = []
# prompt_no_neigh = []
# LLAMA_neigh = []
# LLAMA_no_neigh = []
# LLAMA_alone = []


# for ee in range(len(sentences)): 
#     e = embeddings[ee] 
#     print(sentences[ee])

#     start_time = time.time() # check time to get max
#     value, k = torch.max(torch.sum(e*data.x, dim=-1), 0)
#     print(time.time()-start_time)
#     print(value)
#     sim_score.append(value.item())

    
#     main_text = text_nodeID[k.item()]

#     short_main_text = ' '.join(main_text.split(' ')[:6])

#     neighbors = data.edge_index[1, data.edge_index[0] == k.item()]
#     edge_type = data.edge_type[data.edge_index[0] == k.item()]
#     # print(neighbors)
#     # print(edge_type)

#     n_text = ''
#     for n in range(len(edge_type)):

#         nitem = neighbors[n].item()
#         nt = ' '.join(text_nodeID[nitem].split(' ')[:6])
#         e = edge_type[n].item()      
#         n_text += f" {short_main_text} is {text_edgeID[e+1]} {nt}."
    

#     # print('------------------')
#     # print(short_main_text)
#     # print(n_text)
#     # print('------------------')
#     # exit()

#     prompt = (
#             f"Provide a detailed prompt for image generation of"
#             f" {sentences[ee]}. It must be possible to directly input your answer into an image generation model for an accurate image. "
#             f"Using information about appearance and environment "
#             f"from this data: {main_text} {n_text}"
#             )
    
    
#     prompt_noN = (
#             f"Provide a detailed prompt for image generation of"
#             f" {sentences[ee]}. It must be possible to directly input your answer into an image generation model for an accurate image. "
#             f"Using information about appearance and environment "
#             f"from this data: {main_text}"
#             )
    
#     # print(prompt)
#     prompt_with_neigh.append(prompt)
#     prompt_no_neigh.append(prompt_noN)
#     # print('------------------')

#     chat_completion = client.chat.completions.create(messages = [{
#                 "role": "user", 
#                 "content": prompt
#                 }], model = 'llama3-70b-8192')

#     LLAMA_neigh.append(chat_completion.choices[0].message.content)

#     chat_completion_noN = client.chat.completions.create(messages = [{
#                 "role": "user", 
#                 "content": prompt_noN
#                 }], model = 'llama3-70b-8192')

#     LLAMA_no_neigh.append(chat_completion_noN.choices[0].message.content)

#     chat_completion_alone = client.chat.completions.create(messages = [{
#                 "role": "user", 
#                 "content": f"Provide a detailed prompt for image generation of {sentences[ee]}. It must be possible to directly input your answer into an image generation model for an accurate image."
#                 }], model = 'llama3-70b-8192')

#     LLAMA_alone.append(chat_completion_alone.choices[0].message.content)



# df['SimScore'] = sim_score
# df['DetailsNodeNeighbors'] = prompt_with_neigh
# df['DetailsNode'] = prompt_no_neigh
# df['LLAMANodeNeighbors'] = LLAMA_neigh
# df['LLAMANode'] = LLAMA_no_neigh
# df['LLAMAUserPrompt'] = LLAMA_alone

# df.to_csv('prompts_LLAMA_graph.csv', index=False)

# exit()








### Creating list for nodes of db 

source_nodes = data.edge_index[0]
non_repeat_source_nodes = torch.unique(source_nodes)

df = pd.DataFrame(non_repeat_source_nodes[3362994-1:3362994].numpy())


UserPrompt = []
prompt_with_neigh = []
prompt_no_neigh = []
LLAMA_neigh = []
LLAMA_no_neigh = []
LLAMA_alone = []


for k in non_repeat_source_nodes[3362994-1:3362994]:
    print(k)

    main_text = text_nodeID[k.item()]

    short_main_text = ' '.join(main_text.split(' ')[:6])
    print(short_main_text)

    UserPrompt.append(short_main_text)

    neighbors = data.edge_index[1, data.edge_index[0] == k.item()]
    edge_type = data.edge_type[data.edge_index[0] == k.item()]
    # print(neighbors)
    # print(edge_type)

    n_text = ''
    for n in range(len(edge_type)):

        nitem = neighbors[n].item()
        nt = ' '.join(text_nodeID[nitem].split(' ')[:6])
        e = edge_type[n].item()      
        n_text += f" {short_main_text} is {text_edgeID[e+1]} {nt}."
    

    # print('------------------')
    # print(short_main_text)
    # print(n_text)
    # print('------------------')
    # exit()

    prompt = (
            f"Provide a detailed prompt for image generation of"
            f" {short_main_text}. It must be possible to directly input your answer into an image generation model for an accurate image. "
            f"Using information about appearance and environment "
            f"from this data: {main_text} {n_text}"
            )
    prompt_noN = (
            f"Provide a detailed prompt for image generation of"
            f" {short_main_text}. It must be possible to directly input your answer into an image generation model for an accurate image. "
            f"Using information about appearance and environment "
            f"from this data: {main_text}"
            )

    prompt_with_neigh.append(prompt)
    prompt_no_neigh.append(prompt_noN)

    chat_completion = client.chat.completions.create(messages = [{
                "role": "user", 
                "content": prompt
                }], model = 'llama3-70b-8192')

    LLAMA_neigh.append(chat_completion.choices[0].message.content)

    chat_completion_noN = client.chat.completions.create(messages = [{
                "role": "user", 
                "content": prompt_noN
                }], model = 'llama3-70b-8192')

    LLAMA_no_neigh.append(chat_completion_noN.choices[0].message.content)

    chat_completion_alone = client.chat.completions.create(messages = [{
                "role": "user", 
                "content": f"Provide a detailed prompt for image generation of {short_main_text}. It must be possible to directly input your answer into an image generation model for an accurate image."
                }], model = 'llama3-70b-8192')

    LLAMA_alone.append(chat_completion_alone.choices[0].message.content)


df['UserPrompt'] = UserPrompt
df['DetailsNodeNeighbors'] = prompt_with_neigh
df['DetailsNode'] = prompt_no_neigh
df['LLAMANodeNeighbors'] = LLAMA_neigh
df['LLAMANode'] = LLAMA_no_neigh
df['LLAMAUserPrompt'] = LLAMA_alone

df.to_csv('3362994.csv', index=False)

exit()


    





### Getting individual node and neighbors, generating prompt based on that

# source_nodes = data.edge_index[0]
# non_repeat_source_nodes = torch.unique(source_nodes)

# # print(len(non_repeat_source_nodes))
# for k in non_repeat_source_nodes:

#     main_text = text_nodeID[k.item()]

#     short_main_text = ' '.join(main_text.split(' ')[:6])

#     neighbors = data.edge_index[1, data.edge_index[0] == k.item()]
#     edge_type = data.edge_type[data.edge_index[0] == k.item()]
#     # print(neighbors)
#     # print(edge_type)

#     n_text = ''
#     for n in range(len(edge_type)):

#         nitem = neighbors[n].item()
#         nt = ' '.join(text_nodeID[nitem].split(' ')[:6])
#         e = edge_type[n].item()      
#         n_text += f" {short_main_text} is {text_edgeID[e+1]} {nt}."
    

#     # print('------------------')
#     # print(short_main_text)
#     # print(n_text)
#     # print('------------------')
#     # exit()

#     prompt = (
#             f"Provide a detailed prompt for image generation of"
#             f" {short_main_text}. It must be possible to directly input your answer into an image generation model for an accurate image. "
#             f"Using information about appearance and environment "
#             f"from this data: {main_text} {n_text}"
#             )
    
#     # print(prompt)

#     chat_completion = client.chat.completions.create(messages = [{
#                 "role": "user", 
#                 "content": prompt
#                 }], model = 'llama3-70b-8192')

#     print('chat completed')
#     print(chat_completion.choices[0].message.content)


#     exit()
    












