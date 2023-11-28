import traci
import os

# start the TraCI connection
cur_dir = os.getcwd()
sumo_config_path = os.path.join(cur_dir,"osm.sumocfg")

# start the TraCI connection
traci.start(["sumo", "-c", sumo_config_path])

# Simulate for a certain number of steps
for i in range(50):
    traci.simulationStep()

traci.close()