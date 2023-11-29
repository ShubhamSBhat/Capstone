import traci
import os
import csv

# start the TraCI connection
cur_dir = os.getcwd()
sumo_config_path = os.path.join(cur_dir,"osm.sumocfg")

# start the TraCI connection
traci.start(["sumo", "-c", sumo_config_path])
lane_ids = set()
# Simulate for a certain number of steps
for i in range(400):
    traci.simulationStep()
    vehicle_ids = traci.vehicle.getIDList()
    for vehicle_id in vehicle_ids:
        lane = traci.vehicle.getLaneID(vehicle_id)
        lane_ids.add(lane)
print(lane_ids)
lane_ids = [list(lane_ids)]
csv_file_path = "lane_ids.csv"
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["lane_ids"])
    # Write the header if needed
    # csv_writer.writerow(['ID', 'Name', 'Age'])

    # Write the data
    csv_writer.writerows(lane_ids)
print(len(lane_ids))
traci.close()