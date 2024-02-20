import os
import os.path as osp
import sys
from typing import Callable, List, Optional
import pdb
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class Dataset(InMemoryDataset):



    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(root + '/processed/data_v3.pt')

