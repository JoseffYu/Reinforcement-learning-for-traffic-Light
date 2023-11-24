
import torch
import torch.nn.functional as F
from SUMOenv import SumoEnv
from itertools import count
from stable_baselines3.common.env_checker import check_env


sumo_cfg = "/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Simulator/RunSimulator.sumocfg"

def test(env, wrap_action_in_list=False):
    print(env)

    state = env.reset()
    over = False
    step = 0

    while not over:
        action = env.action_space()
        if wrap_action_in_list:
            action = [action]
        
        action = 0

        next_state, reward, over, info = env.step('J1',action)
        state = env.observation_space()

        if step % 20 == 0:
            print(f"next state: {next_state}")
            #print(torch.from_numpy(state))
            #print(step, state, action, reward,info)

        if step > 200:
            break

        step += 1

print(test(SumoEnv(sumo_cfg, 99999)))