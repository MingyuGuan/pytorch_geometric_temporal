from tqdm import tqdm

import torch
import torch.nn.functional as F
from temporal_gnns import ChebConvLSTM

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader, EnglandCovidDatasetLoader, MontevideoBusDatasetLoader, WikiMathsDatasetLoader, WindmillOutputLargeDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--reuse', action='store_true',
                        help="enable optimization of resusing message passing") 
parser.add_argument('--dataset', type=str, default='CP',
                        help="dataset CP for Chickenpox; COVID for EnglandCovid; BUS for MontevideoBus; WIKI for WikiMaths; WIND for WindmillOutputLarge") 
args = parser.parse_args()

if args.dataset == 'CP':
    loader = ChickenpoxDatasetLoader()
elif args.dataset == 'COVID':
    loader = EnglandCovidDatasetLoader()
elif args.dataset == 'BUS':
    loader = MontevideoBusDatasetLoader()
elif args.dataset == 'WIKI':
    loader = WikiMathsDatasetLoader()
elif args.dataset == "WIND":
    loader = WindmillOutputLargeDatasetLoader()

dataset = loader.get_dataset()

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.9)

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, reuse):
        super(RecurrentGCN, self).__init__()
        self.recurrent = ChebConvLSTM(node_features, 64, 1, reuse=reuse)
        self.linear = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index, edge_weight, h, c):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0

device = torch.device('cuda')

model = RecurrentGCN(node_features=8, reuse=args.reuse).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for epoch in tqdm(range(5)):
    cost = 0
    h, c = None, None
    for time, snapshot in enumerate(train_dataset):
        snapshot.to(device)
        y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    
model.eval()
cost = 0
for time, snapshot in enumerate(test_dataset):
    snapshot.to(device)
    y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
cost = cost / (time+1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))
