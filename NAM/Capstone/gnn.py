import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data


class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GNNLayer, self).__init__(
            aggr='mean')  # 'mean' aggregation method
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GNNLayer(input_dim, hidden_dim)
        self.conv2 = GNNLayer(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# Example data for a single node connected to multiple other nodes
# Waiting time features for the connected nodes
node_features = torch.tensor([[0.5], [0.3], [0.2]])
# Edges from the node to its connected nodes
edge_index = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)

data = Data(x=node_features, edge_index=edge_index.t().contiguous())

input_dim = 1  # Number of input features
hidden_dim = 16  # Hidden layer size
output_dim = 1  # Output size, representing reduced waiting time

model = GNNModel(input_dim, hidden_dim, output_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()  # Mean Squared Error loss

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    output = model(data)
    # Minimize the difference from the original waiting times
    loss = criterion(output, node_features)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# After training, you can use the model to predict reduced waiting times for the connected nodes
with torch.no_grad():
    new_waiting_times = model(data)
    print("Predicted Reduced Waiting Times:", new_waiting_times)
