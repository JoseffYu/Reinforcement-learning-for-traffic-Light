import traci
import sumo

# Connect to SUMO
traci.start(["sumo-gui", "-b","0", "-e","999999", "-c", "/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Simulator/RunSimulator.sumocfg"], numRetries=10,verbose = True)

# Get TrafficLight status
#traffic_light_id = traci.trafficlight.getIDList()
lanes = traci.lane.getIDList()
#lanes_list = ['-E0_0', '-E0_1', '-E0_2', '-E1_0', '-E1_1', '-E1_2', '-E2_0', '-E2_1', '-E2_2', '-E3_0', '-E3_1', '-E3_2', ':J1_0_0', ':J1_10_0', ':J1_11_0', ':J1_1_0', ':J1_2_0', ':J1_3_0', ':J1_4_0', ':J1_5_0', ':J1_6_0', ':J1_7_0', ':J1_8_0', ':J1_9_0', 'E0_0', 'E0_1', 'E0_2', 'E1_0', 'E1_1', 'E1_2', 'E2_0', 'E2_1', 'E2_2', 'E3_0', 'E3_1', 'E3_2']

print(f"lanes:{lanes}")
while traci.simulation.getMinExpectedNumber() > 0:
    # Get TrafficLight status
    traffic_light_id = traci.trafficlight.getIDList()
    #print(f"traffic_light_id:{traffic_light_id}")
    traffic_lanes_id = traci.trafficlight.getControlledLanes(traffic_light_id[0])
    #print(traffic_lanes_id)
    print(traci.trafficlight.getPhaseName(traffic_light_id[0]))
    #print(traci.trafficlight.Phase)
    #print(traci.trafficlight.getAllProgramLogics(traffic_light_id[0])[0].phases)
    if traci.simulation.getTime() == 10:
        traci.trafficlight.setPhase(traffic_light_id[0],4)
    #tls_state = traci.trafficlight.getRedYellowGreenState(traffic_light_id)
    
    # Print TrafficLight Status
    #print("Traffic Light State:", tls_state)
    #print(traci.simulation.getTime())
    traci.simulationStep()
    
traci.close()
