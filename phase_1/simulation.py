import traci
import csv
import pandas as pd
import networkx as nx



# Visualize the graph.
import matplotlib.pyplot as plt

# start the TraCI connection
traci.start(["sumo", "-c", "osm.sumocfg"])
from collections import defaultdict

threshold = 3
x = defaultdict(list)

# with open('output.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow([ 'VID','Speed','Position(X)', 'Position(Y)','LID'])
# simulate for a certain number of steps
for i in range(200):
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
        
        # print("Vehicle ID:", id)
        # print("Speed:", speed)
        # print("Position:", position)
        # print("Lane ID:", lane)

    with open('output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([ 'VID','Speed','Position(X)', 'Position(Y)','LID','max','allowed'])

        for i,j in x.items():
            writer.writerow([i, j[0], j[1],j[2],j[3]])


    # Load data into a Pandas DataFrame.
    data = pd.read_csv('output.csv')

    # Create empty graph.
    G = nx.Graph()

    # Add nodes to graph.
    for i, row in data.iterrows():
        G.add_node(row['VID'], pos=(row['Position(X)'], row['Position(Y)']), speed=row['Speed'], lane=row['LID'])

    # Connect nodes that are within a short distance of each other. 
    for u, u_data in G.nodes(data=True):
        for v, v_data in G.nodes(data=True):
            if u == v:
                continue
            dist = ((u_data['pos'][0]-v_data['pos'][0])**2 + (u_data['pos'][1]-v_data['pos'][1])**2)**0.5
            if dist < 1000:  # Adjust this threshold as needed.
                G.add_edge(u, v)    

     #loop through all the nodes in the graph and check the number of edges
    for id in G.nodes():
        num_edges = len(G.edges(id))
        if num_edges < threshold:
            speed = traci.vehicle.getSpeed(id)
            # increase the speed of the vehicle by 10%
            traci.vehicle.setSpeed(id, speed*1.1)
    
                # writer.writerow([id, speed, position[0],position[1],lane])
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos=pos, with_labels=True)
plt.show()  



# with open('output.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow([ 'VID','Speed','Position(X)', 'Position(Y)','LID'])

#     for i,j in x.items():
#         writer.writerow([i, j[0], j[1],j[2],j[3]])



# end the TraCI connection
traci.close()






