import traci
import numpy as np

# Connect to SUMO
sumo_cfg = "/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Simulator/RunSimulator.sumocfg"
traci.start(["sumo-gui", "-b","0", "-e","999999", "-c", sumo_cfg], numRetries=20,verbose = True)

# Get TrafficLight status
traffic_light_id = traci.trafficlight.getIDList()
lanes = traci.lane.getIDList()
traffic_junc_list = traci.junction.getIDList()

lanes_list = ['-E1_0', '-E1_1', '-E1_2', 'E2_0', 'E2_1', 'E2_2', 'E3_0', 'E3_1', 'E3_2', 'E0_0', 'E0_1', 'E0_2']

print(f"lanes:{len(lanes)}")
#print(f"traffic light:{traffic_light_id}")
step = 0
lanes_length = {lane_id: traci.lane.getLength(lane_id) for lane_id in lanes_list}

def getLanesDensity():
    lanes_density_with_id = {}
    vehicle_size_min_gap = 7.5
    for lane in lanes_list:
        lane_density = min(1, traci.lane.getLastStepVehicleNumber(lane) / (lanes_length[lane] / vehicle_size_min_gap))
        lanes_density_with_id[lane] = lane_density
    return lanes_density_with_id


def getLanesQueue():
    """Returns the queue [0,1] of the vehicles in the incoming lanes of the intersection.

    Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
    """
    #for lane_id in self.lanes_id:
            #self.dict_lane_veh[lane_id] = self.getObservation()
            
    lanes_queue_with_id = {}
    for lane in lanes_list:
        lanes_queue = traci.lane.getLastStepHaltingNumber(lane) / (lanes_length[lane] / max(traci.lane.getLastStepLength(lane),1)) # getLastStepLength Returns the mean vehicle length in m for the last time step on the given lane.

        lanes_queue_with_id[lane] = lanes_queue
    return lanes_queue_with_id


def getObservation():
        
    density = getLanesDensity()
    queue = getLanesQueue()
    lanes_id_obs = {}
    for lane in lanes_list:
        lanes_id_obs[lane] = (density[lane]+queue[lane])/2
    observation = lanes_id_obs
    return observation

    
while step < 1000:
    dict_lane_veh = {}
    #for lane_id in lanes_list:
            #dict_lane_veh[lane_id] = getObservation()
    
    #print(f"observations{getObservation()}")
    
    # Get TrafficLight status
    traffic_light_id = traci.trafficlight.getIDList()
    #print(f"traffic_light_id:{traffic_light_id}")
    traffic_lanes_id = traci.trafficlight.getControlledLanes(traffic_light_id[0])
    #print(traci.trafficlight.Phase)
    #print(traci.trafficlight.getAllProgramLogics(traffic_light_id[0])[0].phases)
    all_green_phases = [phase for phase in traci.trafficlight.getAllProgramLogics(traffic_light_id[0])[0].phases if 'g' in phase.state]
    #print(all_green_phases)
    #if traci.simulation.getTime() == 10:
        #traci.trafficlight.setPhase(traffic_light_id[0],5) #remember here is choosing phase from [0,5] but not name of phase
    #tls_state = traci.trafficlight.getRedYellowGreenState(traffic_light_id)
    if step % 100 == 0:
        time = traci.simulation.getCurrentTime()
        print("current time: ",time)
        print("--------------------------")
    #print("Traffic Light State:", tls_state)
    #print(traci.simulation.getTime())
    traci.simulationStep()
    step += 1
    
traci.close()
