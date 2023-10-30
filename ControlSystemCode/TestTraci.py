import traci

# 连接到SUMO仿真
traci.start(["sumo-gui", "-b","0", "-e","999999", "-c", "/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Simulator/RunSimulator.sumocfg"], numRetries=10,verbose = True)

# 获取红绿灯状态
traffic_light_id = traci.trafficlight.getIDList()
for _ in range(3600):  # 进行3600步仿真，即持续1小时
    traci.simulationStep()
traci.close()
