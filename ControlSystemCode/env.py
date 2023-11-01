import gym
from gym import spaces
import traci
import numpy as np

class SumoEnv(gym.Env):
    def __init__(self, sumo_cfg_file):
        super(SumoEnv, self).__init__()
        self.sumo_cfg_file = sumo_cfg_file
        self.sumo_cmd = ["sumo", "-c", self.sumo_cfg_file]
        self.action_space = spaces.Discrete(8)   #Action Space, need to be changed, decision made by TrafficLight
        # State Space, Need to be changed, need to contain both Light phase and vehicle things 
        self.observation_space = spaces.Box(low=0, high=10000, shape=(5,))  
        self.step_count = 0

    def reset(self):
        # reset SUMO
        traci.start(self.sumo_cmd)
        self.step_count = 0
        # get initial observation
        return self._get_observation()

    #TO DO:
    def step(self, action):
        next_state = None
        reward = None
        done = False
        info = {'do_action': None}
        # start calculate reward
        start = False
        
        #To do:
        do_action = self.traffic_signal.change_phase(action)  #Add traffic light control
        if do_action is None:
            return next_state, reward, done, info

        traci.simulationStep()
        
        #self things need to be modified
        if do_action == -1 and self.change_action_time is None:
            self.change_action_time = traci.simulation.getTime()
        
        #To do:
        if self.change_action_time is not None and self.sumo.simulation.getTime() >= self.change_action_time:
            self.change_action_time = None
            self.train_state = self._compute_state()
            start = True

        #To do:
        # compute_state must be front of compute_reward
        next_state = self._compute_next_state()
        reward = self._compute_reward(start, do_action)
        done = self._compute_done()
        info = {'do_action': do_action}
        self.step_count += 1
        return next_state, reward, done, info

    #To do:
    def _take_action(self, action):
        # Take action to control Traffic Light
        pass
    #To do:
    def _get_observation(self):
        # Get state status
        pass

    def close(self):
        # Close SUMO
        traci.close()
