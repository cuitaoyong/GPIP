import torch
import torch.nn as nn
from PygMD17 import MD17
from dig.threedgraph.method import SchNet,SphereNet
from run import run
from dig.threedgraph.evaluation import ThreeDEvaluator
import numpy as np
import pdb
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set the root directory, data name and checkpoint directory
root = './'
data_name = 'benzene2017'
checkpoint_path_pth = "./checkpoint/pretraining_schnet_2.pth"

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

# Load the MD17 dataset
dataset = MD17(root='./', name=data_name)

# Split the dataset into train, validation, and test sets
split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=42)



train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
test_dataset = test_dataset[:1000]


# Create an instance of the SchNet model
model = SchNet(energy_and_force=True, cutoff=5.0, num_layers=6, hidden_channels=128, out_channels=1, num_filters=128, num_gaussians=50)

# model = SphereNet(energy_and_force=True, cutoff=5.0, num_layers=4, 
#         hidden_channels=128, out_channels=1, int_emb_size=64, 
#         basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256, 
#         num_spherical=3, num_radial=6, envelope_exponent=5, 
#         num_before_skip=1, num_after_skip=2, num_output_layers=3 
#         )

# # Update the model's linear layer
model.update_u.lin2 = nn.Linear(128//2, 100)
# model.update_vs[3].lin = torch.nn.Linear(256,100)
# model.update_vs[2].lin = torch.nn.Linear(256,100)
# model.update_vs[1].lin = torch.nn.Linear(256,100)
# model.update_vs[0].lin = torch.nn.Linear(256,100)
# model.init_v.lin = torch.nn.Linear(256,100)

# Load the pre-trained model weights from the checkpoint
checkpoint_path = torch.load(checkpoint_path_pth)
model.load_state_dict(checkpoint_path)

# Change the model's linear layer to output 1 channel
model.update_u.lin2 = nn.Linear(128//2, 1)
# model.update_vs[3].lin = torch.nn.Linear(256,1)
# model.update_vs[2].lin = torch.nn.Linear(256,1)
# model.update_vs[1].lin = torch.nn.Linear(256,1)
# model.update_vs[0].lin = torch.nn.Linear(256,1)
# model.init_v.lin = torch.nn.Linear(256,1)

# Define the loss function and evaluation metric
loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

# Create an instance of the run class
run3d = run()

# Run the training process
run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation,
          epochs=1000, batch_size=2, vt_batch_size=64, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=150, p=100, energy_and_force=True)