from tqdm import tqdm

import torch
import torch.nn.functional as F
from temporal_gnns import ChebConvGRU, ChebConvAgg

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader, WindmillOutputSmallDatasetLoader,WindmillOutputMediumDatasetLoader,WindmillOutputLargeDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--reuse', action='store_true',
                        help="enable optimization of resusing message passing")  
args = parser.parse_args()

device = torch.device('cuda')

# loader = ChickenpoxDatasetLoader()
loader = WindmillOutputLargeDatasetLoader()

dataset = loader.get_dataset()

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.9)

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, reuse):
        super(RecurrentGCN, self).__init__()
        self.recurrent = ChebConvGRU(node_features, 64, 1, reuse=reuse)
        self.linear = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
        
model = RecurrentGCN(node_features = 8, reuse=args.reuse).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for epoch in tqdm(range(5)):
    cost = 0
    for time, snapshot in enumerate(train_dataset):
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
    snapshot.to(device)
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
cost = cost / (time+1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))
