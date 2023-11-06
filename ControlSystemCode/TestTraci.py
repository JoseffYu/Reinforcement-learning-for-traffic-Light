import traci
import sumo

# Connect to SUMO
traci.start(["sumo-gui", "-b","0", "-e","999999", "-c", "/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Simulator/RunSimulator.sumocfg"], numRetries=10,verbose = True)

# Get TrafficLight status
#traffic_light_id = traci.trafficlight.getIDList()
lanes = traci.lane.getIDList()
#lanes_list = ['-E12_0', '-E12_1', '-E12_2', '-E13_0', '-E13_1', '-E13_2', '-E14_0', '-E14_1', '-E14_2', '-E15_0', '-E15_1', '-E15_2', ':J13_0_0', ':J13_10_0', ':J13_11_0', ':J13_1_0', ':J13_2_0', ':J13_3_0', ':J13_4_0', ':J13_5_0', ':J13_6_0', ':J13_7_0', ':J13_8_0', ':J13_9_0', 'E12_0', 'E12_1', 'E12_2', 'E13_0', 'E13_1', 'E13_2', 'E14_0', 'E14_1', 'E14_2', 'E15_0', 'E15_1', 'E15_2']

print(f"lanes:{lanes}")
while traci.simulation.getMinExpectedNumber() > 0:
    # Get TrafficLight status
    traffic_light_id = traci.trafficlight.getIDList()
    print(f"traffic_light_id:{traffic_light_id}")
    traffic_lanes_id = traci.trafficlight.getControlledLanes(traffic_light_id[0])
    print(traffic_lanes_id)
    print(traci.trafficlight.getPhase(traffic_light_id[0]))
    if traci.simulation.getTime() == 10:
        traci.trafficlight.setPhase(traffic_light_id[0],3)
    #tls_state = traci.trafficlight.getRedYellowGreenState(traffic_light_id)
    
    # Print TrafficLight Status
    #print("Traffic Light State:", tls_state)
    print(traci.simulation.getTime())
    traci.simulationStep()
    
traci.close()
