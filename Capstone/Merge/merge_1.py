import traci
import os

# start the TraCI connection
cur_dir = os.getcwd()
sumo_config_path = os.path.join(cur_dir,"merge.sumocfg")

EV=1
AV=2
HV=3
# start the TraCI connection
traci.start(["sumo", "-c", sumo_config_path])
avg_waiting_time=0
avg_depart_delay=0
avg_speed=0
overall_waiting_time=0
count=0
count_2=0

# Simulate for a certain number of steps
for i in range(300):
    traci.simulationStep()
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
    for vehicle_id in vehicle_ids:
        # Extract relevant attributes
        vehicle_type = vehicle_type_mapping[vehicle_id]
        
        # Add the vehicle data to the dictionary
        overall_waiting_time +=  traci.vehicle.getWaitingTime(vehicle_id)/10
        count_2+=1
        if(vehicle_type==1):
            count+=1
            waiting_time =traci.vehicle.getWaitingTime(vehicle_id)
            avg_speed+= traci.vehicle.getSpeed(vehicle_id)
            print(avg_speed)
            depart_delay =traci.vehicle.getDepartDelay(vehicle_id)
            avg_waiting_time = avg_waiting_time+waiting_time
            
            avg_depart_delay= avg_depart_delay+depart_delay
avg_depart_delay = avg_depart_delay
avg_waiting_time= avg_waiting_time
overall_waiting_time = overall_waiting_time
print("Emergency waiting time:",avg_waiting_time)
print("Emergency average speed: ",avg_speed*10/count)
print("Emergency depart delay: ",avg_depart_delay)
print("Overall Waiting time: ",overall_waiting_time)
print(traci.simulation.getCollidingVehiclesNumber())
traci.close()