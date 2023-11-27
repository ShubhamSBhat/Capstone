import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv
import csv
import matplotlib.pyplot as plt




# Create a PyTorch Geometric Data object




# Define the GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        # x = nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Define a simple neural network for link prediction
class LinkPredictionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinkPredictionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # print(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        # return torch.sigmoid(x)
        return x

# Combine GraphSAGE and Neural Network for link prediction
class GraphSAGENet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGENet, self).__init__()
        self.graphsage = GraphSAGE(input_dim, hidden_dim, output_dim)
        self.nn = LinkPredictionNN(output_dim, hidden_dim, 1)

    def forward(self, data):
        x = self.graphsage(data)
        edge_index = data.edge_index
        edge_embeddings = x[edge_index[0],:] * x[edge_index[1],:]
        edge_embeddings = edge_embeddings.view(edge_embeddings.size(0), -1)
        link_predictions = self.nn(edge_embeddings)
        # print(link_predictions)
        return link_predictions

# Custom collate function for PyTorch Geometric Data objects
def custom_collate(batch):
    return Data(x=torch.cat([item.x for item in batch], dim=0),
                edge_index=torch.cat([item.edge_index for item in batch], dim=1),
                y=torch.cat([item.y for item in batch], dim=0))

if __name__ == "_main_":
    csv_file_path = 'edge_list.csv'
    edge_list = []

    # Reading from CSV
    with open(csv_file_path, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        for row in csv_reader:
            row = [int(i) for i in row]
            edge_list.append(row)


    vehicle_features = []
    csv_file_path = "vehicle_features.csv"

    with open(csv_file_path, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            row = [float(i) for i in row]
            vehicle_features.append(row)

    csv_file_path = "target_y.csv"
    target_y = []
    with open(csv_file_path, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            row = [float(i) for i in row]
            target_y.append(row)

    target_y = target_y[0]


    x = torch.tensor(vehicle_features, dtype=torch.float)
    # Instantiate the model
    input_dim = 2  # Assuming each node has a feature vector of size 3
    hidden_dim = 10
    output_dim = 5
    model = GraphSAGENet(input_dim, hidden_dim, output_dim)

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    n = len(target_y)
    train_n = int(n * (3 / 4))
    train_edge_index = [edge_list[0][:train_n + 1],edge_list[1][:train_n+1]]
    train_target_y = target_y[:train_n+1]

    test_edge_index = [edge_list[0][train_n+1:], edge_list[1][train_n+1:]]
    test_target_y = target_y[train_n+1:]

    
    train_edge_index = torch.tensor(train_edge_index,dtype=torch.long)
    test_edge_index = torch.tensor(test_edge_index,dtype=torch.long)

    train_target_y = torch.tensor(train_target_y,dtype=torch.float).view(-1,1)
    test_target_y = torch.tensor(test_target_y,dtype=torch.float).view(-1,1)

    train_data = Data(x=x, edge_index=train_edge_index, y=train_target_y)
    test_data = Data(x=x, edge_index=test_edge_index, y=test_target_y)

    # Convert the data to a batch using DataLoader with custom collate function
    train_dataset = [train_data]
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=custom_collate)

    test_dataset = [test_data]
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=custom_collate)

    # print(train_edge_index)
    train_losses = []
    test_losses = []

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            train_predictions = model(batch)
            print(train_predictions, batch.y)
            train_loss = criterion(train_predictions, batch.y)
            train_losses.append(train_loss.item())
            test_predictions = model(test_dataset[0])
            test_loss = criterion(test_predictions, test_dataset[0].y)
            test_losses.append(test_loss.item())

            print("epoch:",epoch)
            print("===================")
            print("loss:",train_loss.item(),test_loss.item())
            
            train_loss.backward()
            optimizer.step()

    # Test the model
        
    epochs = 100
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), "GraphSAGENet.pth")