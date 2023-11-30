from threading import Thread
from time import sleep
import numpy as np
import traci
import csv
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

# Define the GNN model
class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
# Define a custom policy for taking actions based on node embeddings
def custom_action_policy(node_embeddings, vehicle_ids):
    for idx, id in enumerate(vehicle_ids):
        # You can define your custom action policy here based on node embeddings
        # For example, you can calculate a score based on node embeddings and take actions accordingly
        score = node_embeddings[idx][0]  # Using the first dimension of the embedding as an example
        if score > 0.5:
            # Increase speed if the score is above a threshold
            speed = traci.vehicle.getSpeed(id)
            traci.vehicle.setSpeed(id, speed * 1.1)

# start the TraCI connection
traci.start(["sumo", "-c", "osm.sumocfg"])
from collections import defaultdict

# Create the GNN model
gnn_model = GNNModel(in_channels=5, hidden_channels=32, out_channels=16)  # Adjust input and output dimensions as needed
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)

# Simulate for a certain number of steps
for i in range(300):
    sleep(0.1)
    traci.simulationStep()

    # Get a list of all the vehicles in the simulation
    vehicle_ids = traci.vehicle.getIDList()
    
    # Create an empty graph
    G = nx.Graph()

    # Loop through all the vehicles and get their information
    for id in vehicle_ids:
        speed = traci.vehicle.getSpeed(id)
        position = traci.vehicle.getPosition(id)
        lane = traci.vehicle.getLaneID(id)
        neigh = traci.vehicle.getMaxSpeed(id)
        l = traci.vehicle.getAllowedSpeed(id)
        traci.vehicle.
        # Add a node to the graph for each vehicle
        G.add_node(id, pos=position, speed=[speed, neigh, l], lane=lane)

        # Connect nodes that are within a short distance of each other
    for u, u_data in G.nodes(data=True):
        for v, v_data in G.nodes(data=True):
            if u == v:
                continue
            dist = ((u_data['pos'][0]-v_data['pos'][0])**2 + (u_data['pos'][1]-v_data['pos'][1])**2)**0.5
            if dist < 1000:  # Adjust this threshold as needed.
                G.add_edge(u, v)

    # Check if there are edges in the graph
    if G.number_of_edges() > 0:

        # Prepare data for the GNN
        node_features = []  # Store vehicle features as node features
        # Create a mapping from vehicle IDs to numerical indices
        id_to_index = {id: index for index, id in enumerate(G.nodes())}

        # Initialize an empty list to store edge indices
        edge_index = []

        for id in G.nodes():
            speed_tuple = G.nodes[id]['speed']
            position_tuple = G.nodes[id]['pos']
            node_features.append(list(speed_tuple) + list(position_tuple))
            for neighbor in G.neighbors(id):
                edge_index.append([id_to_index[id], id_to_index[neighbor]])

        x = torch.tensor(node_features, dtype=torch.float)
        
   
        # Convert the list of edge indices into a NumPy array
        edge_index = np.array(edge_index, dtype=np.long)

        # Transpose the edge_index array
        edge_index = torch.from_numpy(edge_index).t().contiguous()

        # Forward pass through the GNN
        node_embeddings = gnn_model(x, edge_index)
        print(node_embeddings)
        # Take action based on custom policy and node embeddings
        custom_action_policy(node_embeddings, vehicle_ids)
    else:
        # Handle the case where there are no edges in the graph
        pass

    # ...




# Save the data to a CSV file
with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['VID', 'Speed', 'Position(X)', 'Position(Y)', 'LID'])

    for id in G.nodes():
        pos = G.nodes[id]['pos']
        speed = G.nodes[id]['speed']
        lane = G.nodes[id]['lane']
        writer.writerow([id, speed, pos[0], pos[1], lane])

# End the TraCI connection
traci.close()
