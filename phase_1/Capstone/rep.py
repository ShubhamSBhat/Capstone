import traci
import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import pandas as pd

# Path to your SUMO configuration file
cur_dir = os.getcwd()
sumo_config_path = os.path.join(cur_dir, "Simulation", "test.sumocfg")

# Start the SUMO simulation using TraCI
traci.start(["sumo", "-c", sumo_config_path])

# Initialize a graph for the network representation
G = nx.Graph()
threshold = 3
x = defaultdict(list)

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    # get a list of all the vehicles in the simulation
    vehicle_ids = traci.vehicle.getIDList()

    # loop through all the vehicles and get their information
    for id in vehicle_ids:
        speed = traci.vehicle.getSpeed(id)
        position = traci.vehicle.getPosition(id)
        lane = traci.vehicle.getLaneID(id)
        neigh = traci.vehicle.getMaxSpeed(id)
        l = traci.vehicle.getAllowedSpeed(id)
        a = traci.vehicle.getAcceleration(id)
        x[id] = [speed, position[0], position[1], lane, neigh, l, a]
        print("Vehicle ID:", id)
        print("Speed:", speed)
        print("Position:", position)
        print("Lane ID:", lane)

    with open('output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['VID', 'Speed', 'Position(X)',
                        'Position(Y)', 'LID', 'max', 'allowed', 'Acceleration'])

        for i, j in x.items():
            writer.writerow([i, j[0], j[1], j[2], j[3], j[4], j[5], j[6]])

    # Load data into a Pandas DataFrame.
    data = pd.read_csv('output.csv')

    # Create empty graph.
    G.clear()

    # Add nodes to graph.
    for i, row in data.iterrows():
        G.add_node(row['VID'], pos=(row['Position(X)'],
                   row['Position(Y)']), speed=row['Speed'], lane=row['LID'])

    # Connect nodes that are within a short distance of each other.
    for u, u_data in G.nodes(data=True):
        for v, v_data in G.nodes(data=True):
            if u == v:
                continue
            dist = ((u_data['pos'][0]-v_data['pos'][0])**2 +
                    (u_data['pos'][1]-v_data['pos'][1])**2)**0.5
            if dist < 50:  # Adjust this threshold as needed.
                G.add_edge(u, v)

# Draw the graph with a beautiful and clear layout
plt.figure(figsize=(10, 8))
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=3000, font_size=12, font_color='black',
        node_color='lightblue', edge_color='gray', width=2, alpha=0.7)

plt.title("Graph Representation of Vehicle Positions")
plt.show()

# Close the TraCI connection
traci.close()
