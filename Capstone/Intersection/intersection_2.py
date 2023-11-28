from collections import defaultdict
from threading import Thread
from time import sleep
import numpy as np
import traci
import csv
import pandas as pd
import networkx as nx
import torch
import os



import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader

EV = 1
AV = 2
HV = 3

lanes = [":J1_0_0",":J1_1_0",":J1_1_1","E0_0","E0_1","E4_0","E4_1","E7_0"]
lane_encoder = LabelEncoder()
encoded_lanes = lane_encoder.fit_transform(lanes)
lanes_mapping = {}
for i,lane in enumerate(lanes):
    lanes_mapping[lane] = encoded_lanes[i]


# Define the custom policy for taking actions based on node embeddings
def calculate_score( vehicle_type, speed, x,y, lane, neighbors):
    score = 0.0

    # Consider vehicle type: Give a higher score to EVs
    # Justification: Encouraging the adoption of EVs by providing them with preferential treatment in terms of lane changes and speed adjustments can promote a more sustainable and environmentally friendly transportation system.

    if vehicle_type == "EV":
        score += 0.2
        print("Vehicle type: EV (score increased by 0.2)")

    # Consider proximity to other vehicles: Penalize vehicles that are too close
    # Justification: Maintaining a safe distance from other vehicles is crucial for preventing accidents and ensuring smooth traffic flow. Penalizing vehicles that are too close incentivizes them to maintain a safe distance, reducing the risk of collisions.
    
    if neighbors!=None:
        
        X,Y= traci.vehicle.getPosition(neighbors)
        distance = get_distance(x,y,X,Y )
        if distance < 500:
            score -= 0.1
            print("Vehicle proximity: Too close (score decreased by 0.1)")

    # Consider current speed: Penalize vehicles that are already going fast
    # Justification: Encouraging moderate speeds can help regulate traffic flow and reduce the risk of accidents. Penalizing vehicles that are already going fast incentivizes them to slow down, promoting a more controlled and safe driving environment.

    if speed > 20:
        score -= 0.1
        print("Current speed: Exceeding limit (score decreased by 0.1)")

    # Consider lane position: Give a higher score to vehicles in the middle lane
    # Justification: Vehicles in the middle lane are generally less likely to encounter congestion or lane changes, making it a more efficient and predictable route. Giving a higher score to vehicles in the middle lane encourages them to maintain their position, promoting smoother traffic flow.

    if lane == "center":
        score += 0.1
        print("Lane position: Center (score increased by 0.1)")

    return score

def custom_action_policy(graph, vehicle_data):
    # print("in custom action")
    for idx, id in enumerate(vehicle_data.keys()):
        vehicle_type = vehicle_type_mapping[vehicle_id]
        speed = vehicle_data[vehicle_id]['speed']
        x = vehicle_data[vehicle_id]['x']
        y = vehicle_data[vehicle_id]['y']
        lane = vehicle_data[vehicle_id]['lane']
        neighbors=vehicle_data[vehicle_id]['neighbor']

        # Calculate a score based on node attributes and graph information
        score = calculate_score(vehicle_type, speed, x,y, lane, neighbors)

        # Take action based on the calculated score
        if score > 0.5:
            # Increase speed if the score is above a threshold
            new_speed = speed * 1.1
            adjust_speed(graph, id, new_speed)
        elif score < -0.5:
            # Decrease speed if the score is below a threshold
            new_speed = speed * 0.8
            adjust_speed(graph, id, new_speed)

        # Implement lane-changing behavior based on graph information
        if vehicle_type == "EV":
            if has_AV_in_same_lane_in_graph(graph, id, lane):
                # Initiate lane change if there is an AV in the same lane
                change_lane_to_outer_lane(graph, id)
            else:
                # Check for potential lane change opportunities
                if lane != "center":
                    if can_change_lane_to_center_in_graph(graph, id, lane):
                        # Change lane to the center if possible
                        change_lane_to_center_lane(graph, id)
                else:
                    if can_change_lane_to_outer_in_graph(graph, id, lane):
                        # Change lane to an outer lane if necessary
                        change_lane_to_outer_lane(graph, id)

def get_distance(x,y, X,Y):
    """Calculates the Euclidean distance between two positions."""
    
    # print(position1)
    # print(position2)
    distance = np.sqrt((x - X)*2 + (y - Y)*2)

    return distance
# Define functions for lane-changing actions
def adjust_speed(graph, vehicle_id, new_speed):
    # Adjust vehicle speed in the simulation
    traci.vehicle.setSpeed(vehicle_id, new_speed)

    # Update the speed attribute in the graph
    graph.nodes[vehicle_id]['speed'] = new_speed

def get_adjacent_lanes(graph, vehicle_id, lane):
    """Identifies the adjacent lanes to the given vehicle's lane."""
    adjacent_lanes = []

    # Determine the adjacent lanes based on the current lane
    if lane == "left":
        adjacent_lanes.append("center")
    elif lane == "center":
        adjacent_lanes.append("left")
        adjacent_lanes.append("right")
    elif lane == "right":
        adjacent_lanes.append("center")

    # Check if there are any vehicles in the adjacent lanes
    for adjacent_lane in adjacent_lanes:
        for neighbor in graph.neighbors(vehicle_id):
            neighbor_attributes = graph.nodes[neighbor]
            neighbor_lane = neighbor_attributes['lane']
            if neighbor_lane == adjacent_lane:
                adjacent_lanes.remove(adjacent_lane)

    return adjacent_lanes

def change_lane_to_center_lane(graph, vehicle_id):
    # Initiate lane change to the center lane
    traci.vehicle.changeLaneRelative(vehicle_id, 1)

    # Update the lane attribute in the graph
    new_lane = traci.vehicle.getLaneID(vehicle_id)
    graph.nodes[vehicle_id]['lane'] = new_lane

def change_lane_to_outer_lane(graph, vehicle_id):
    # Initiate lane change to an outer lane
    traci.vehicle.changeLaneRelative(vehicle_id, -1)

    # Update the lane attribute in the graph
    new_lane = traci.vehicle.getLaneID(vehicle_id)
    graph.nodes[vehicle_id]['lane'] = new_lane

def can_change_lane_to_center_in_graph(graph, vehicle_id, lane):
    # Check if there are any vehicles in the adjacent lanes
    adjacent_lanes = get_adjacent_lanes(graph, vehicle_id, lane)
    for adjacent_lane in adjacent_lanes:
        if lane != "center" and adjacent_lane == "center":
            return True

    return False

def can_change_lane_to_outer_in_graph(graph, vehicle_id, lane):
    # Check if there are any vehicles in the adjacent lanes
    adjacent_lanes = get_adjacent_lanes(graph, vehicle_id, lane)
    if len(adjacent_lanes) == 1 and lane != "center":
        return True

    return False

def has_AV_in_same_lane_in_graph(graph, vehicle_id, lane):
    # Check if there are any AVs in the same lane
    for neighbor in graph.neighbors(vehicle_id):
        neighbor_attributes = graph.nodes[neighbor]
        neighbor_type = neighbor_attributes['vehicle_type']
        neighbor_lane = neighbor_attributes['lane']

        if neighbor_type == "AV" and neighbor_lane == lane:
            return True

    return False

def predict_link(vehicle_features,edge_list):
    global model
    x = torch.tensor(vehicle_features,dtype=torch.float)
    edge_index = torch.tensor(edge_list, dtype=torch.long)
    dataset = [Data(x=x, edge_index=edge_index)]
    return model(dataset[0])
    

def transform_data(vehicle_data):
    vehicle_ids = list(vehicle_data.keys())
    # print(vehicle_ids)
    vehicle_ids.append(None)
    vehicle_ids_encoder = LabelEncoder()
    encoded_vehicle_ids = vehicle_ids_encoder.fit_transform(vehicle_ids)
    vehicle_ids_mapping = {}
    for i,vehicle_id in enumerate(vehicle_ids):
        vehicle_ids_mapping[vehicle_id] = encoded_vehicle_ids[i]
    vehicle_features = []
    for veh_id, features in vehicle_data.items():
        # print(veh_id,features)
        feature_list = [vehicle_ids_mapping[veh_id],features['x'],features['y']]
        vehicle_features.append(feature_list)

    vehicle_features.sort(key = lambda x:x[0])
    vehicle_features = [feature_list[1:] for feature_list in vehicle_features]

    return vehicle_features,vehicle_ids_mapping

def create_traffic_graph(vehicle_data, threshold_distance):
    G = nx.Graph()
    
    # Add nodes for each vehicle
    for vehicle_id in vehicle_data:
        vehicle_type = vehicle_type_mapping[vehicle_id]
        speed = vehicle_data[vehicle_id]['speed']
        x = vehicle_data[vehicle_id]['x']
        y = vehicle_data[vehicle_id]['y']
        lane = vehicle_data[vehicle_id]['lane']
        acceleration = vehicle_data[vehicle_id]['acceleration']
        neighbor = vehicle_data[vehicle_id]['neighbor']
        G.add_node(vehicle_id, vehicle_type=vehicle_type, speed=speed, x=x,y=y, lane=lane,
                  acceleration=acceleration,neighbor =neighbor)
    
    
    # Connect vehicles based on distance difference and lane information
    
    if len(vehicle_data)>1:
        vehicle_ids = list(vehicle_data.keys())
        for i in range(len(vehicle_ids)):
            vehicle_id = vehicle_ids[i]
            for j in range(i+1,len(vehicle_ids)):
                other_vehicle_id = vehicle_ids[j]
                # print(other_vehicle_id, "this prints id")
                if other_vehicle_id != vehicle_id:
                    X = vehicle_data[other_vehicle_id]['x']
                    Y = vehicle_data[other_vehicle_id]['x']
                    distance = get_distance(x,y,X,Y)
                    other_lane = vehicle_data[other_vehicle_id]['lane']

                    if distance < threshold_distance and lane == other_lane:
                        G.add_edge(vehicle_id, other_vehicle_id)

                
    return G
        
# start the TraCI connection
cur_dir = os.getcwd()
sumo_config_path = os.path.join(cur_dir,"intersection.sumocfg")

# start the TraCI connection
traci.start(["sumo", "-c", sumo_config_path])

# Simulate for a certain number of steps
for i in range(100):
    traci.simulationStep()
# Create a mapping to classify vehicles as AV or HV
    vehicle_type_mapping = {}

# Get a list of all the vehicles in the simulation
    vehicle_ids = traci.vehicle.getIDList()

    for id in vehicle_ids:
        if id.startswith("route0") or id.startswith("route1"):
            vehicle_type_mapping[id] = EV
        elif id.startswith("route4"):
            vehicle_type_mapping[id] = AV
        else:
            vehicle_type_mapping[id] = HV
    
    vehicle_data = {}

    for vehicle_id in vehicle_ids:
        # Extract relevant attributes
        vehicle_type = vehicle_type_mapping[vehicle_id]
        speed = int(traci.vehicle.getSpeed(vehicle_id))
        x,y = traci.vehicle.getPosition(vehicle_id)
        lane = traci.vehicle.getLaneID(vehicle_id)
        acceleration = traci.vehicle.getAcceleration(vehicle_id)
        neighbor = traci.vehicle.getLeader(vehicle_id)
        if neighbor is not None:
            neighbor = neighbor[0]
        # Add the vehicle data to the dictionary
        vehicle_data[vehicle_id] = {
            'vehicle_type': vehicle_type,
            'speed': speed,
            'x':x,
            'y':y,
            'lane': lane,
            'acceleration': acceleration,
            'neighbor':neighbor
        }
        
        
        # Define the threshold distance for forming edges between vehicles
        threshold_distance = 1000

        # Construct the graph using the `create_traffic_graph` function
        G = create_traffic_graph(vehicle_data, threshold_distance)
        
        # Check if there are edges in the graph
        
        if G.number_of_edges() > 0:
            # Take action based on custom policy and graph information
            custom_action_policy(G, vehicle_data)
        else:
            # Handle the case where there are no edges in the graph
            pass

 
# Save the data to a CSV file
with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['VID', 'Speed', 'Position(X)', 'Position(Y)', 'LID'])

    for id in G.nodes():
        x = G.nodes[id]['x']
        y = G.nodes[id]['y']
        speed = G.nodes[id]['speed']
        lane = G.nodes[id]['lane']
        writer.writerow([id, speed, x, y, lane])

traci.close()