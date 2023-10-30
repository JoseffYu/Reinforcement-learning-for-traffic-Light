import traci
import SUMOEnv

#traci.start(["sumo-gui", "-b","0", "-e","3600", "-n", "/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Simulator/Road.net.xml","-r", "/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Simulator/Route.rou.xml"], numRetries=12000,verbose = True)

SUMOEnv.env.runSUMO()
second = 0
while second < 3600:
    SUMOEnv.env.step()
    second += 1
traci.close()