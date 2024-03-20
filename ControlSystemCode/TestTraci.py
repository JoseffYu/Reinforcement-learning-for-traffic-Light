import traci
import numpy as np

# Connect to SUMO
sumo_cfg = "/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Simulator/ComplexEnv.sumocfg"
traci.start(["sumo-gui", "-b","0", "-e","999999", "-c", sumo_cfg], numRetries=20,verbose = True)

# Get TrafficLight status
traffic_light_id = traci.trafficlight.getIDList()
lanes = traci.lane.getIDList()
traffic_junc_list = traci.junction.getIDList()

#print(f"lanes:{len(lanes)}")
print(f"traffic light:{traffic_light_id}")



step = 0

    
while step < 1000:
    dict_lane_veh = {}
    #for lane_id in lanes_list:
            #dict_lane_veh[lane_id] = getObservation()
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
    print(traci.trafficlight.getAllProgramLogics(traffic_light_id[0])[0].phases)
    #print("Traffic Light State:", tls_state)
    #print(traci.simulation.getTime())
    traci.simulationStep()
    step += 1
    
traci.close()
