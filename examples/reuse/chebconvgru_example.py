from tqdm import tqdm

import torch
import torch.nn.functional as F
from temporal_gnns import ChebConvGRU

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader, MTMDatasetLoader, MontevideoBusDatasetLoader, WikiMathsDatasetLoader, WindmillOutputLargeDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

import argparse
from torch_geometric.data import Data, Batch

parser = argparse.ArgumentParser()
parser.add_argument('--reuse', action='store_true',
                        help="enable optimization of resusing message passing") 
parser.add_argument('--dataset', type=str, default='CP',
                        help="dataset CP for Chickenpox; HAND for MTM Hand Motions; BUS for MontevideoBus; WIKI for WikiMaths; WIND for WindmillOutputLarge") 
parser.add_argument('--in-feats', type=int, default=8, help="num of node features")
parser.add_argument('--epochs', type=int, default=10, help="num of epochs")
parser.add_argument('--rep', type=int, default=1, help="Relicate nodes for scalability test; 1 for original dataset")
parser.add_argument('--num-layers', type=int, default=1, help="number of GNN-RNN layers")
args = parser.parse_args()

if args.dataset == 'CP':
    loader = ChickenpoxDatasetLoader()
elif args.dataset == 'WIKI':
    loader = WikiMathsDatasetLoader()
elif args.dataset == "WIND":
    loader = WindmillOutputLargeDatasetLoader()
# elif args.dataset == 'HAND':
#     loader = MTMDatasetLoader() 
#     node_features = 3 # failed
# elif args.dataset == 'BUS':
#     loader = MontevideoBusDatasetLoader() 
#     node_features = 16 # failed
# elif args.dataset == 'COVID':
#     loader = MontevideoBusDatasetLoader() # Too small

dataset = loader.get_dataset(lags = args.in_feats)

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.9)

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, num_layers, reuse):
        super(RecurrentGCN, self).__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(ChebConvGRU(node_features, 64, 1, reuse=reuse))
        for _ in range(self.num_layers-1):
            self.layers.append(ChebConvGRU(64, 64, 1, reuse=reuse))
        # self.reccurrent = ChebConvGRU(node_features, 64, 1, reuse=reuse)
        self.linear = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index, edge_weight):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_weight)
        # h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

device = torch.device('cuda')

model = RecurrentGCN(node_features=args.in_feats, num_layers=args.num_layers, reuse=args.reuse).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for epoch in tqdm(range(args.epochs)):
    cost = 0
    for time, snapshot in enumerate(train_dataset):
        if args.rep > 1:
            snapshot = Batch.from_data_list([Data(x=snapshot.x, edge_index=snapshot.edge_index, edge_attr=snapshot.edge_attr, y=snapshot.y) for i in range(args.rep)])
        snapshot.to(device)
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    
model.eval()
cost = 0
for time, snapshot in enumerate(test_dataset):
    if args.rep > 1:
        snapshot = Batch.from_data_list([Data(x=snapshot.x, edge_index=snapshot.edge_index, edge_attr=snapshot.edge_attr, y=snapshot.y) for i in range(args.rep)])
    snapshot.to(device)
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
cost = cost / (time+1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))
