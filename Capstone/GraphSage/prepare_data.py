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
import csv 

EV = 1
AV = 2
HV = 3

vehicle_dataset,edge_list = None,None
target_y = None
edge_count = 0

lanes = []
csv_file_path = "lane_ids.csv"
with open(csv_file_path, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        for row in csv_reader:
            lanes.append(row)
        
        lanes = lanes[1]

lane_encoder = LabelEncoder()
encoded_lanes = lane_encoder.fit_transform(lanes)
lanes_mapping = {}
for i,lane in enumerate(lanes):
    lanes_mapping[lane] = encoded_lanes[i]

def get_distance(x,y, X,Y):
    """Calculates the Euclidean distance between two positions."""
    
    # print(position1)
    # print(position2)
    distance = np.sqrt((x - X)*2 + (y - Y)*2)

    return distance


def create_traffic_graph(vehicle_data, threshold_distance):
    G = nx.Graph()
    current_edge_count = 0
    edge_list = [[-1],[-1]]
    target_y = []
    # Add nodes for each vehicle
    for vehicle_id in vehicle_data:
        vehicle_type = vehicle_type_mapping[vehicle_id]
        speed = vehicle_data[vehicle_id]['speed']
        x = vehicle_data[vehicle_id]['x']
        y = vehicle_data[vehicle_id]['y']
        # lane = vehicle_data[vehicle_id]['lane']
        acceleration = vehicle_data[vehicle_id]['acceleration']
        
        G.add_node(vehicle_id, vehicle_type=vehicle_type, speed=speed, x=x,y=y, lane=lane,
                  acceleration=acceleration)
    # print(vehicle_data)
    vehicle_ids = list(vehicle_data.keys())
    # vehicle_ids.append(None)
    vehicle_ids_encoder = LabelEncoder()
    encoded_vehicle_ids = vehicle_ids_encoder.fit_transform(vehicle_ids)
    vehicle_ids_mapping = {}
    for i,vehicle_id in enumerate(vehicle_ids):
        vehicle_ids_mapping[vehicle_id] = encoded_vehicle_ids[i]
    # Connect vehicles based on distance difference and lane information
    if len(vehicle_data)>1:
        for i in range(len(vehicle_ids)):
            vehicle_id = vehicle_ids[i]
            for j in range(i+1,len(vehicle_ids)):
                other_vehicle_id = vehicle_ids[j]
                # print(other_vehicle_id, "this prints id")
                
                if other_vehicle_id != vehicle_id:
                    X = vehicle_data[other_vehicle_id]['x']
                    Y = vehicle_data[other_vehicle_id]['x']
                    edge_list[0].append(vehicle_ids_mapping[vehicle_id])
                    edge_list[1].append(vehicle_ids_mapping[other_vehicle_id])
                    distance = get_distance(x,y,X,Y)
                    # other_lane = vehicle_data[other_vehicle_id]['lane']

                    if distance < threshold_distance:
                        target_y.append(1)
                        G.add_edge(vehicle_id, other_vehicle_id)
                        current_edge_count += 1
                    else:
                        target_y.append(0)
    # global edge_count
    # global vehicle_dataset
    
    # if current_edge_count > edge_count:
    #     vehicle_dataset = vehicle_data.copy()
    #     edge_count = current_edge_count
        
    edge_list[0].pop(0)
    edge_list[1].pop(0)
    return G,edge_list,target_y,current_edge_count
        
# start the TraCI connection
cur_dir = os.getcwd()
sumo_config_path = os.path.join(cur_dir, "osm.sumocfg")

# start the TraCI connection
traci.start(["sumo", "-c", sumo_config_path])

# Simulate for a certain number of steps
for i in range(30):
    traci.simulationStep()
# Create a mapping to classify vehicles as AV or HV
    vehicle_type_mapping = {}
    print("step:",i)
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
        # vehicle_type = vehicle_type_mapping[vehicle_id]
        speed = int(traci.vehicle.getSpeed(vehicle_id))
        x,y = traci.vehicle.getPosition(vehicle_id)
        lane = lanes_mapping[traci.vehicle.getLaneID(vehicle_id)]
        acceleration = traci.vehicle.getAcceleration(vehicle_id)
        neighbor = traci.vehicle.getLeader(vehicle_id)
        if neighbor is not None:
            neighbor = neighbor[0]
        # Add the vehicle data to the dictionary
        vehicle_data[vehicle_id] = {
            # 'vehicle_type': vehicle_type,
            'speed': speed,
            'x':x,
            'y':y,
            'lane': lane,
            'acceleration': acceleration
        }
        
        
        # Define the threshold distance for forming edges between vehicles
        threshold_distance = 1000

        # Construct the graph using the `create_traffic_graph` function
        G,current_edge_list,current_target_y,current_edge_count = create_traffic_graph(vehicle_data, threshold_distance)
        if current_edge_count > edge_count:
            edge_count = current_edge_count
            vehicle_dataset = vehicle_data.copy()
            edge_list = current_edge_list.copy()
            target_y = current_target_y.copy()
            print(target_y)


traci.close()
vehicle_ids = list(vehicle_dataset.keys())
vehicle_ids.append(None)
vehicle_ids_encoder = LabelEncoder()
encoded_vehicle_ids = vehicle_ids_encoder.fit_transform(vehicle_ids)
vehicle_ids_mapping = {}
for i,vehicle_id in enumerate(vehicle_ids):
    vehicle_ids_mapping[vehicle_id] = encoded_vehicle_ids[i]

vehicle_features = []

for veh_id, features in vehicle_dataset.items():
    # print(veh_id,features)
    feature_list = [vehicle_ids_mapping[veh_id],features['x'],features['y'],features['lane'],features['speed'],features['acceleration']]
    vehicle_features.append(feature_list)

vehicle_features.sort(key = lambda x:x[0])
vehicle_features = [feature_list[1:] for feature_list in vehicle_features]


print(edge_list) 

csv_file_path = "vehicle_features.csv"
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write the header if needed
    # csv_writer.writerow(['ID', 'Name', 'Age'])

    # Write the data
    csv_writer.writerows(vehicle_features)

csv_file_path = "edge_list.csv"
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write the header if needed
    # csv_writer.writerow(['ID', 'Name', 'Age'])

    # Write the data
    csv_writer.writerows(edge_list)

csv_file_path = "target_y.csv"
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write the header if needed
    # csv_writer.writerow(['ID', 'Name', 'Age'])

    # Write the data
    csv_writer.writerows([target_y])



