
import gym
import traci
import numpy as nps
import os
import sys
import random
from MltiTLEnv import TrafficLight
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
import sumolib


class SumoEnv(gym.Env):
    def __init__(
        self,
        sumo_cfg_file:str,
        simulation_time:int,
        use_gui: bool = False
        ):
        super(SumoEnv, self).__init__()
        self.sumo_cfg_file = sumo_cfg_file
        self.sumo_cmd = ["sumo-gui", "-c", self.sumo_cfg_file]
        self.time = 0
        self.end_time = simulation_time
        self.use_gui = use_gui
        self.sumo = None
        self.sumoBinary = 'sumo-gui'
        if self.use_gui:
            self.sumoBinary = 'sumo-gui'
        self.start_sumo(simulation_time)
        conn = traci
        
        self.tl_ids = traci.trafficlight.getIDList()
        self.traffic_light = TrafficLight(self.tl_ids,traci=conn, simulation_time=self.end_time)
        
        self.close()


    def start_sumo(self, simulation_time):
        # simulation_time = 100
        traci.start(["sumo", "-b", "0", "-c", self.sumo_cfg_file], numRetries=10) #add -gui with sumo if visualization is needed
        traci.simulationStep(simulation_time)

    def reset(self): #def reset(self, simulation_time):
        # reset SUMO
        traci.start(["sumo", "-b", "0", "-c", self.sumo_cfg_file], numRetries=10) #add -gui with sumo if visualization is needed
        #traci.simulationStep(simulation_time) #active it if visualisation is needed
        self.time = 0
        self.traffic_light.reward = 0
        # get initial observation
        return self.observation_space()


    def step(self, tl_ids, action):
        next_state = None
        reward = None
        done = False
        info = {'do_action': None}
        for id in tl_ids:
            do_action = self.traffic_light.doAction(id, action)
            if do_action is None:
                return next_state, reward, done, info

            traci.simulationStep()
            # compute_state must be front of compute_reward
        
            next_state = self.computeNextState()
            reward = self.computeReward(id,do_action)
            done = self.computeDone()
            info = {'do_action': do_action}
            self.time += 1
        return self.computeState(),next_state, reward, done, info

    
    def render(self, mode='human'):
        pass


    def takeAction(self, tl_id ,action):
        # Take action to control Traffic Light
        return self.traffic_light.doAction(tl_id,action)
    
    
    def computeReward(self, tl_id, do_action):
        tl_reward = self.traffic_light.computeReward(tl_id,do_action)
        ts_total_reward = self.traffic_light.total_reward
        return tl_reward
    
    
    def computeDone(self):
        current_time = traci.simulation.getTime()
        if current_time >= self.end_time:
            done = True
        else:
            done = False
        return done
    
    
    def computeState(self):
        return self.traffic_light.computeState()
    
    
    def computeNextState(self):
        return self.traffic_light.computeNextState()


    def close(self):
        # Close SUMO
        traci.close()


    def observation_space(self):
        # Get state status
        return self.traffic_light.observation_space
    

    def action_space(self):
        return self.traffic_light.action_space
