import argparse
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch.optim as optim
import random
from tqdm import tqdm
import numpy as np
from model import GNNDecoder
from net3d import Net3D
import dgl
from loss import *
from Dataset import Dataset
from SchNet import SchNet
from SphereNet import SphereNet
import pdb

# Function to perturb the input data
def perturb(positions, mu, sigma):

    device = positions.device
    positions_perturb = positions + torch.normal(mu, sigma, size=positions.size()).to(device)

    return positions_perturb

def mask(x, num_atom_type, mask_rate):

    

    num_atoms = x.size()[0]
    sample_size = int(num_atoms * mask_rate + 1)
    masked_atom_indices = random.sample(range(num_atoms), sample_size)
    mask_node_labels_list = []
    for atom_idx in masked_atom_indices:
        mask_node_labels_list.append(x[atom_idx].view(1, -1))
    mask_node_label = torch.cat(mask_node_labels_list, dim=0)
    masked_atom_indices = torch.tensor(masked_atom_indices)

    atom_type = F.one_hot(mask_node_label[:, 0], num_classes=num_atom_type).float()
    # data.node_attr_label = torch.cat((atom_type,atom_chirality), dim=1)
    node_attr_label = atom_type

    # modify the original node feature of the masked node
    for atom_idx in masked_atom_indices:
        x[atom_idx] = torch.tensor([num_atom_type])

    x_perturb = x

    return x_perturb,node_attr_label,masked_atom_indices
# Function to train the model
def train_mae(args, model_list, loader, optimizer_list, device, loss2, model3d, alpha_l=1.0, loss_fn="sce"):
    # Set the loss criterion based on the loss function argument
    if loss_fn == "sce":
        criterion = partial(sce_loss, alpha=alpha_l)
    else:
        criterion = nn.CrossEntropyLoss()

    model, dec_pred_atoms = model_list
    optimizer_model, optimizer_dec_pred_atoms, optimizer_3d = optimizer_list
    
    model.train()
    model3d.train()
    dec_pred_atoms.train()


    loss_accum = 0

    epoch_iter = tqdm(loader, desc="Iteration")
    for step, batch in enumerate(epoch_iter):

        batch = batch.to(device)

        Z = batch.x[:, 0]

        edge_index = batch.edge_index

        pos_noise = perturb(batch.pos, 0, 0.15)

        x_perturb, node_attr_label, masked_atom_indices = mask(Z, num_atom_type=args.NUM_NODE_ATTR, mask_rate=args.mask_rate)

        node_rep, decoder, pred_noise = model(x_perturb, pos_noise, batch.batch)

        test = torch.zeros(1, 256)
        batch_num_nodes= []
        all_num_nodes = 0
        for i in range(max(batch.batch) + 1):
            k = (batch.batch == i).sum(dim=0)
            batch_num_nodes.append(k)
            src, dst = get_pairwise(k, device=device)
            g = dgl.graph((src, dst), device=device)
            g.ndata['feat'] = Z[all_num_nodes:all_num_nodes + k]
            g.ndata['x'] = batch.pos[all_num_nodes:all_num_nodes + k]
            g.edata['d'] = torch.norm(g.ndata['x'][g.edges()[0]] - g.ndata['x'][g.edges()[1]], p=2, dim=-1).unsqueeze(
                -1).detach() 
            all_num_nodes += k
            node_rep21 = model3d(g)
            if i != 0:
                test = torch.cat((test, node_rep21), 0)
            else:
                test = node_rep21
          
        batch_num_nodes = torch.tensor(batch_num_nodes).to(device)
    
        b = torch.zeros_like(batch.edge_attr)
        edge_attr = torch.stack((batch.edge_attr, b), 1).squeeze(-1)   

        # Compute losses

        pred_node = dec_pred_atoms(decoder, edge_index, edge_attr, masked_atom_indices)
        dm_loss = criterion(node_attr_label, pred_node[masked_atom_indices])
        threed_loss = loss2(node_rep, test, nodes_per_graph=batch_num_nodes)
        de_loss = F.mse_loss(
            pred_noise, pos_noise - batch.pos, reduction='sum'
        )
        
        loss = dm_loss + args.loss_ratio * threed_loss + de_loss

        optimizer_model.zero_grad()
        optimizer_dec_pred_atoms.zero_grad()
        optimizer_3d.zero_grad()

        loss.backward()

        optimizer_3d.step()
        optimizer_model.step()
        optimizer_dec_pred_atoms.step()

        loss_accum += float(loss.cpu().item())
        epoch_iter.set_description(f"train_loss: {loss.item():.4f}")

    return loss_accum / step 

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--path', type=str, default="./Pretrain_dataset/",
                        help='input dataset path')
    parser.add_argument('--epochs', type=int, default=3,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--NUM_NODE_ATTR', type=int, default=9,
                        help='define the number of node attributes')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0.01,
                        help='weight decay (default: 0)')
    parser.add_argument('--emb_dim', type=int, default=100,
                        help='embedding dimensions (default: 100)')
    parser.add_argument('--loss_ratio', type=float, default=0.1,
                        help='3D Net ratio')
    parser.add_argument('--mask_rate', type=float, default=0.10,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--output_model_file', type=str, default = './checkpoint/', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="linear")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--input_model_file', type=str, default=None)
    parser.add_argument("--alpha_l", type=float, default=1.0)
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--model", type=str, default="SchNet")
    parser.add_argument("--use_scheduler", action="store_true", default=True)
    args = parser.parse_args()

    print(args)

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Set the device (GPU or CPU)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Set the dataset path
    path = args.path


    # Create the dataset and data loader objects
    dataset = Dataset(path)
    loader = DataLoader(dataset, args.batch_size, shuffle=True)

    # Initialize the models and move them to the device
    if args.model == SchNet:
        
        model = SchNet(energy_and_force=True, cutoff=5.0, num_layers=6, hidden_channels=128, out_channels=1, num_filters=128, num_gaussians=50).to(device)
        model.update_u.lin2 = nn.Linear(128//2,args.emb_dim).to(device)

    else:
        model = SphereNet(energy_and_force=True, cutoff=5.0, num_layers=4, 
                hidden_channels=128, out_channels=1, int_emb_size=64, 
                basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256, 
                num_spherical=3, num_radial=6, envelope_exponent=5, 
                num_before_skip=1, num_after_skip=2, num_output_layers=3 
                )
        model.update_vs[3].lin = torch.nn.Linear(256,args.emb_dim)
        model.update_vs[2].lin = torch.nn.Linear(256,args.emb_dim)
        model.update_vs[1].lin = torch.nn.Linear(256,args.emb_dim)
        model.update_vs[0].lin = torch.nn.Linear(256,args.emb_dim)
        model.init_v.lin = torch.nn.Linear(256,args.emb_dim)
    
    model3d = Net3D(node_dim=0, edge_dim=1,hidden_dim=20,target_dim=args.emb_dim,hidden_edge_dim=20,node_wise_output_layers=0,message_net_layers=1,update_net_layers=1,reduce_func="mean",fourier_encodings=4,propagation_depth=1,dropout=0.0,batch_norm=True,readout_batchnorm=True,batch_norm_momentum=0.93,readout_hidden_dim=20,readout_layers=1,readout_aggregators=["min","max","mean"],
              avg_d=1).to(device)
    atom_pred_decoder = GNNDecoder(args.emb_dim, args.NUM_NODE_ATTR, JK="last", gnn_type=args.gnn_type).to(device)
      
    model_list = [model, atom_pred_decoder] 
  
    # Initialize the optimizers for each model
    optimizer_model = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_atoms = optim.AdamW(atom_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_3d = optim.AdamW(model3d.parameters(), lr=args.lr, weight_decay=args.decay)
    
    optimizer_list = [optimizer_model, optimizer_dec_pred_atoms, optimizer_3d]

    if args.use_scheduler:
        print("--------- Use scheduler -----------")
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.epochs) ) * 0.5
        scheduler_model = torch.optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=scheduler)
        scheduler_dec = torch.optim.lr_scheduler.LambdaLR(optimizer_dec_pred_atoms, lr_lambda=scheduler)
        scheduler_list = [scheduler_model, scheduler_dec, None]
    else:
        scheduler_model = None
        scheduler_dec = None  
      
  
    output_file_temp = args.output_model_file
  
    # Training loop
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
          
        loss2 = NTXent()
          
        train_loss = train_mae(args, model_list, loader, optimizer_list, device, alpha_l=args.alpha_l, loss_fn=args.loss_fn, loss2 = loss2, model3d = model3d)
        
        # Save the model
        torch.save(model.state_dict(), output_file_temp + f"pretraining_{args.model}_{epoch}.pth")
        print("saved")

        print("train_loss:",train_loss)
        if scheduler_model is not None:
            scheduler_model.step()
        if scheduler_dec is not None:
            scheduler_dec.step()

if __name__ == "__main__":
    main()
