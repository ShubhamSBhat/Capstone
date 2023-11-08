import traci
import csv
  import matplotlib.pyplot as plt
# start the TraCI connection
traci.start(["sumo", "-c", "osm.sumocfg"])
from collections import defaultdict


x = defaultdict(list)

# with open('output.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow([ 'VID','Speed','Position(X)', 'Position(Y)','LID'])
# simulate for a certain number of steps
for i in range(1500):
    traci.simulationStep()

    # get a list of all the vehicles in the simulation
    vehicle_ids = traci.vehicle.getIDList()

    # loop through all the vehicles and get their information
    for id in vehicle_ids:
        speed = traci.vehicle.getSpeed(id)
        position = traci.vehicle.getPosition(id)
        lane = traci.vehicle.getLaneID(id)
        neigh = traci.vehicle.getMaxSpeed(id)
        l =traci.vehicle.getAllowedSpeed(id)
        x[id] = [speed,position[0],position[1],lane,neigh,l]
        
        print("Vehicle ID:", id)
        print("Speed:", speed)
        print("Position:", position)
        print("Lane ID:", lane)
        print("Neighbour: ", neigh)
        print("Allowed: ", l)
                # writer.writerow([id, speed, position[0],position[1],lane])

print(x)


with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([ 'VID','Speed','Position(X)', 'Position(Y)','LID','max','allowed'])

    for i,j in x.items():
        writer.writerow([i, j[0], j[1],j[2],j[3],j[4],j[5]])


    # Load data into a Pandas DataFrame.
    data = pd.read_csv('output.csv')

    # Create empty graph.
    G = nx.Graph()
    # Add nodes to graph.
    for i, row in data.iterrows():
        # G.add_node(row['VID'], pos=(row['Position(X)'], row['Position(Y)']), speed=row['Speed'], lane=row['LID'])
        G.add_node(row['VID'], pos=(row['Position(X)'], row['Position(Y)']), speed=row['Speed'])
    # Connect nodes that are within a short distance of each other.
    for u, u_data in G.nodes(data=True):
        for v, v_data in G.nodes(data=True):
            if u == v:
                continue
            dist = ((u_data['pos'][0]-v_data['pos'][0])**2 + (u_data['pos'][1]-v_data['pos'][1])**2)**0.5
            if dist < 500:  # Adjust this threshold as needed.
                G.add_edge(u, v)

    # Visualize the graph.
  

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos=pos, with_labels=True)
    plt.show()


# end the TraCI connection
traci.close()




import pandas as pd
import networkx as nx

# Load data into a Pandas DataFrame.
data = pd.read_csv('output.csv')

# Create empty graph.
G = nx.Graph()
# Add nodes to graph.
for i, row in data.iterrows():
    # G.add_node(row['VID'], pos=(row['Position(X)'], row['Position(Y)']), speed=row['Speed'], lane=row['LID'])
     G.add_node(row['VID'], pos=(row['Position(X)'], row['Position(Y)']), speed=row['Speed'])
# Connect nodes that are within a short distance of each other.
for u, u_data in G.nodes(data=True):
    for v, v_data in G.nodes(data=True):
        if u == v:
            continue
        dist = ((u_data['pos'][0]-v_data['pos'][0])**2 + (u_data['pos'][1]-v_data['pos'][1])**2)**0.5
        if dist < 500:  # Adjust this threshold as needed.
            G.add_edge(u, v)

# Visualize the graph.
import matplotlib.pyplot as plt

pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos=pos, with_labels=True)
plt.show()





# import networkx as nx
# from random import sample

# # Remove some edges at random
# edges_to_remove = sample(G.edges(), int(0.4 * G.number_of_edges()))
# G.remove_edges_from(edges_to_remove)

# # Predict the missing edges using the Common Neighbors method
# potential_edges = []
# for u in G.nodes():
#     for v in G.nodes():
#         if u != v and not G.has_edge(u, v):
#             common_neighbors = set(G.neighbors(u)).intersection(set(G.neighbors(v)))
#             if len(common_neighbors) > 0:
#                 potential_edges.append((u, v))

# # Sort the potential edges by the number of common neighbors
# potential_edges = sorted(potential_edges, key=lambda x: len(set(G.neighbors(x[0])).intersection(set(G.neighbors(x[1])))), reverse=True)

# # Print the top 10 potential edges
# for u, v in potential_edges[:10]:
#     print(f'Node {u} and node {v} should have an edge.')


# true_positives = 0
# false_positives = 0
# false_negatives = 0
# for u, v in edges_to_remove:
#     if G.has_edge(u, v):
#         true_positives += 1
#     else:
#         false_negatives += 1
# for u, v in potential_edges[:10]:
#     if G.has_edge(u, v):
#         true_positives += 1
#     else:
#         false_positives += 1

# # Calculate precision and recall
# precision = true_positives / (true_positives + false_positives)
# recall = true_positives / (true_positives + false_negatives)

# print(f"Precision: {precision:.3f}")
# print(f"Recall: {recall:.3f}")

# import networkx as nx
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score

# train_edges, test_edges = train_test_split(list(G.edges()), test_size=0.2, random_state=42)

# # Calculate the Jaccard similarity for each pair of nodes in the test set
# scores = []
# for u, v in test_edges:
#     u_neighbors = set(G.neighbors(u))
#     v_neighbors = set(G.neighbors(v))
#     jaccard = len(u_neighbors.intersection(v_neighbors)) / len(u_neighbors.union(v_neighbors))
#     scores.append(jaccard)

# # Calculate the ROC AUC score
# y_true = [1] * len(test_edges) + [0] * len(test_edges)
# y_pred = scores + list(reversed(scores))
# roc_auc = roc_auc_score(y_true, y_pred)

# print(f'ROC AUC score: {roc_auc}')