



from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
import base64
import os
import time
import requests
from selenium.webdriver.edge.options import Options
from io import BytesIO
from PIL import Image













def scrape_images_with_edge(query, index, num_images=10):
    # Set up Edge WebDriver

    options = Options()
    options.add_argument("--guest")  # Use a guest profile
    service = Service("C:/Users/annak/Downloads/edgedriver_win64/msedgedriver.exe")  # Replace with your Edge WebDriver path
    # driver = webdriver.Edge(service=service)
    driver = webdriver.Edge(service=service, options=options)

    # Create an output folder
    if not os.path.exists(f"images/{index}"):
        os.mkdir(f"images/{index}")

    try:
        # Open Google Images
        search_query = '+'.join(query.split())
        url = f"https://www.google.com/search?q={search_query}&source=lnms&tbm=isch"
        driver.get(url)

        # Allow the page to load
        time.sleep(3)

        # Find image elements
        images = driver.find_elements(By.CSS_SELECTOR, "img")
        images = images[1:] # skip google et al
        count = 0

        for i, img in enumerate(images):
            if count >= num_images:
                break

            # Get the image source
            img_src = img.get_attribute("src")
            # print(img_src)
            if img_src:
                try:
                    if img_src.startswith("data:image"):
                        # Handle Base64-encoded images
                        header, encoded = img_src.split(",", 1)
                        img_format = header.split(";")[0].split("/")[1]  # Extract format
                        img_data = base64.b64decode(encoded)

                        min_width = 100
                        min_height = 100
                        with Image.open(BytesIO(img_data)) as img:
                            width, height = img.size
                            if width >= min_width and height >= min_height:
                # Save the image if dimensions meet the threshold
                                with open(f"images/{index}/image_{count + 1}.{img_format}", 'wb') as img_file:
                                    img_file.write(img_data)
                                    print(f"Image {count + 1} (Base64 Large: {width}x{height}) saved.")
                                    count += 1
                            else:
                                print(f"Base64 image skipped: {width}x{height} (too small).")
                   
                    
                    elif img_src.startswith("http"):
                        # Handle regular image URLs
                        response = requests.get(img_src)
                        with open(f"images/{index}/image_{i + 1}.jpg", 'wb') as img_file:
                            img_file.write(response.content)
                        print(f"Image {count + 1} (URL) saved.")
                    # count += 1
                except Exception as e:
                    print(f"Failed to save image {i + 1}: {e}")

    finally:
        driver.quit()


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


for k in range(200,400):#indices: #non_repeat_source_nodes[0:100]:
    print(k)

    main_text = text_nodeID[k]#.item()]

    short_main_text = ' '.join(main_text.split(' ')[:6])
    print(short_main_text)

    scrape_images_with_edge(re.sub(r'[^\w\s]','',short_main_text), str(k), 5)

