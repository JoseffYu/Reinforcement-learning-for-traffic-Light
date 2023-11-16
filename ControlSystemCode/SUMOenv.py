
import gym
import traci
import numpy as nps
from TrafficLightEnv import TrafficLight


class SumoEnv(gym.Env):
    def __init__(
        self,
        sumo_cfg_file:str,
        simulation_time:int,
        ):
        super(SumoEnv, self).__init__()
        self.sumo_cfg_file = sumo_cfg_file
        self.sumo_cmd = ["sumo", "-c", self.sumo_cfg_file]
        self.time = 0
        self.end_time = simulation_time
        traci.start(["sumo-gui", "-b","0", "-e",str(simulation_time), "-c", sumo_cfg_file], numRetries=10,verbose = True)
        
        conn = traci
        
        self.tl_id = traci.trafficlight.getIDList()
        self.traffic_light = TrafficLight(self.tl_id,traci=conn)
        
        self.close()


    def reset(self):
        # reset SUMO
        traci.start(self.sumo_cmd)
        self.time = 0
        # get initial observation
        return self.getObservation()


    def step(self, tl_id, action):
        next_state = None
        reward = None
        done = False
        info = {'do_action': None}
        # start calculate reward
        start = False
        
        do_action = self.traffic_light.doAction(tl_id, action)
        if do_action is None:
            return next_state, reward, done, info

        traci.simulationStep()
        # compute_state must be front of compute_reward
        next_state = self.computeNextState()
        reward = self.computeReward(start, do_action)
        done = self.computeDone()
        info = {'do_action': do_action}
        self.time += 1
        return next_state, reward, done, info
    
    
    def render(self, mode='human'):
        pass


    def takeAction(self, action):
        # Take action to control Traffic Light
        return self.traffic_light.doAction(action)
    
    
    def computeReward(self, do_action):
        ts_reward = self.traffic_light.computeReward(do_action)
        return ts_reward
    
    
    def computeDone(self):
        current_time = traci.simulation.getTime()
        if current_time > self.end_time:
            done = True
        else:
            done = False
        return done
    
    
    def computeState(self):
        return self.traffic_light.computeState()
    
    
    def computeNextState(self):
        return self.traffic_light.computeNextState


    def close(self):
        # Close SUMO
        traci.close()


    def observation_space(self):
        # Get state status
        return self.traffic_light.observation_space
    

    def action_space(self):
        return self.traffic_light.action_space
