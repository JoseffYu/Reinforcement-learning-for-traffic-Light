from SUMOenv import SumoEnv
from stable_baselines3.common.env_checker import check_env

sumo_cfg = "/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Simulator/RunSimulator.sumocfg"
env = SumoEnv(sumo_cfg_file=sumo_cfg, simulation_time=99999)
print(check_env(env))
