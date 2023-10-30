import traci

class env():
    
    def runSUMO():
        traci.start(["sumo-gui", "-b","0", "-e","3600", "-c", "/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Simulator/RunSimulator.sumocfg"], numRetries=10,verbose = True)
        return True
        
        
    def step():
        traci.simulationStep()


    def closeSimulator():
        traci.close()
        return True      


    def getTrafficLightIds():
        traffic_light_ids = traci.trafficlight.getIDList()
        return traffic_light_ids


    def setRedLight(light_ids):
        
        traffic_light_ids = env.getTrafficLightIds()
        
        for light in traffic_light_ids:
            traci.trafficlight.setRedYellowGreenState(light, "rrrr")
        
        return True


    def setGreenLight(light_ids):

        traffic_light_ids = env.getTrafficLightIds()
        
        for light in traffic_light_ids:
            traci.trafficlight.setRedYellowGreenState(light, "gggg")
        
        return True
