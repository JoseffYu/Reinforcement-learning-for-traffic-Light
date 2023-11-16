
import torch
import torch.nn.functional as F
from SUMOenv import SumoEnv
from itertools import count
from stable_baselines3.common.env_checker import check_env


sumo_cfg = "/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Simulator/RunSimulator.sumocfg"
device = torch.device("cpu")
env = SumoEnv(sumo_cfg_file=sumo_cfg, simulation_time=99999)

state = env.observation_space()
action_space = env.action_space()
lr = 0.0001


env.close()
