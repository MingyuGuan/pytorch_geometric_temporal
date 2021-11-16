from tqdm import tqdm

import torch
import torch.nn.functional as F
from temporal_gnns import GCNConvLSTM

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader, WindmillOutputSmallDatasetLoader,WindmillOutputMediumDatasetLoader,WindmillOutputLargeDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

import torch_geometric.transforms as T

device = torch.device('cuda')

# loader = ChickenpoxDatasetLoader()
loader = WindmillOutputLargeDatasetLoader()

dataset = loader.get_dataset()

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GCNConvLSTM(node_features, 32)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight, h, c):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0
        
model = RecurrentGCN(node_features=8).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for epoch in tqdm(range(5)):
    cost = 0
    h, c = None, None
    for time, snapshot in enumerate(train_dataset):
        snapshot = T.ToSparseTensor()(snapshot)
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
    snapshot = T.ToSparseTensor()(snapshot)
    snapshot.to(device)
    y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
cost = cost / (time+1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))
