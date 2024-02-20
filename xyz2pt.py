
from sys import argv
import torch
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.data import (
    Data,
    InMemoryDataset,
)
import torch.nn.functional as F
from torch_geometric.nn import radius_graph
from ase.io import read




if __name__ == '__main__':
    file2=argv[1]+".pt"
    import os
    # read from extxyz files
    atoms_all_list=read(filename=argv[1]+".xyz",index=slice(None))

    data_list =[]
    for i, atoms in enumerate(atoms_all_list):
            natom = len(atoms)
            pos_arr = atoms.positions
            pos = pos_arr.tolist()
            atoms_numbers = atoms.numbers.tolist()
            nbr_dist = [[[i, 0.0]] for i in range(natom)]

            # define x as atoms numbers
            x = torch.as_tensor(atoms.numbers, dtype=torch.int)
            b=torch.zeros_like(x)
            #define pos as atomic positions
            pos=torch.as_tensor(atoms.positions, dtype=torch.float)

            # define edge_index and edge_attr using radius_graph
            edge_attr=torch.stack((x,b),1).squeeze(-1) 
            edge_index = radius_graph(pos, r=6.0, batch=torch.zeros_like(x))
            row, col = edge_index
            dist = (pos[row] - pos[col]).norm(dim=-1)
            #save data into Data using torch_geometric.data
            data = Data(
                pos=pos,
                z=x,
                x=edge_attr,
                edge_index = edge_index,
                edge_attr = dist,
                natoms=natom,
            )
            data_list.append(data)

    # save to file
    torch.save(InMemoryDataset.collate(data_list), file2)
