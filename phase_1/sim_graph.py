import traci
import csv
import pandas as pd
import networkx as nx

# start the TraCI connection
traci.start(["sumo", "-c", "osm.sumocfg"])
from collections import defaultdict

# set the threshold for the number of edges per vehicle
threshold = 3

# simulate for a certain number of steps
for i in range(200):
    traci.simulationStep()

    # get a list of all the vehicles in the simulation
    vehicle_ids = traci.vehicle.getIDList()

    # create an empty graph
    G = nx.Graph()

    # loop through all the vehicles and get their information
    for id in vehicle_ids:
        speed = traci.vehicle.getSpeed(id)
        position = traci.vehicle.getPosition(id)
        lane = traci.vehicle.getLaneID(id)
        neigh = traci.vehicle.getMaxSpeed(id)
        l =traci.vehicle.getAllowedSpeed(id)
        # add a node to the graph for each vehicle
        G.add_node(id, pos=position, speed=[speed,neigh,l], lane=lane)

    # connect nodes that are within a short distance of each other
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



    

# save the data to a CSV file
with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['VID', 'Speed', 'Position(X)', 'Position(Y)', 'LID'])

    for id in G.nodes():
        pos = G.nodes[id]['pos']
        speed = G.nodes[id]['speed']
        lane = G.nodes[id]['lane']
        writer.writerow([id, speed, pos[0], pos[1], lane])
# end the TraCI connection
traci.close()
