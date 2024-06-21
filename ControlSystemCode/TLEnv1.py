
import math
import numpy as np
from statistics import mean
from gym import spaces

'''
    Reinforcement learning environment for traffic light agent, this is environment 1
'''

class TrafficLight:
    def __init__(
        self,
        tls_id: str,
        simulation_time: int,
        traci
    ):
        self.conn = traci
        self.tls_id = tls_id
        self.rs_update_time = 0
        self.update_time = True
        self.green_phase = 0 # first action
        self.yellow_phase = None
        self.end_time = 0
        self.all_phases = []
        self.lanes_id = []
        self.dict_action_wait_num = []
        self.correct_action = 0
        self.continue_reward = False
        self.dict_lane_veh = None
        self.simulation_time = simulation_time
        self.total_reward = 0
        self.lanes_density = {}
        self.duration = 0

        for tl_id in self.tls_id:
            num_tl = self.tls_id.index(tl_id)
            self.all_phases.append(self.conn.trafficlight.getAllProgramLogics(tl_id)[0].phases)
            self.lanes_id.append(list(dict.fromkeys(self.conn.trafficlight.getControlledLanes(tl_id))))
            self.lanes_length = {lane_id: self.conn.lane.getLength(lane_id) for lane_id in self.lanes_id[num_tl]}

        self.all_green_phases = []
        for index in range(len(self.all_phases)):
            for phase in self.all_phases[index] :
                if 'g' in phase.state:
                    self.all_green_phases.append(phase)

        self.observation_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.all_green_phases))
        
        self.lanes_id = [lane for sub_list_lanes in self.lanes_id for lane in sub_list_lanes]


    #action is the order of phase
    def doAction(self, tl_id, action):
        self.computeUpdate()
        new_green_phase_state = self.all_green_phases[action].state
        self.conn.trafficlight.setRedYellowGreenState(tl_id, new_green_phase_state)
        if self.update_time == True:
            self.green_phase = action
            return action
        else:
            return self.green_phase
            
            
    def computeReward(self, action):
        update_reward = True
        arrive_vehicles = self.conn.simulation.getArrivedNumber()
        self.last_arrived_vehicles = self.conn.simulation.getArrivedNumber()
        average_density = mean(self.getLanesDensity())
        rewards = arrive_vehicles + self.chooseMinWaitTime(action, update_reward) + self.duration + self.getLanesDensity()
        self.total_reward += rewards
        return rewards
    
    
    def computeUpdate(self):
        current_time = self.conn.simulation.getTime()/1000
        if current_time >= self.end_time:
            self.update_time = True
        else:
            self.update_time = False
    
    
    def chooseMinWaitTime(self,action, update_reward):
        self.dict_lane_veh = {}
        for i,lane_id in enumerate(self.lanes_id):
            self.dict_lane_veh[lane_id] = self.conn.lane.getLastStepHaltingNumber(lane_id)
       
        # merge wait_num by actions, position po each lines indicate a  action of traffic light
        self.dict_action_wait_num = [self.dict_lane_veh['-E1_1'] + self.dict_lane_veh['-W1_1'],
                                self.dict_lane_veh['-N1_2'] + self.dict_lane_veh['-S1_2'],
                                self.dict_lane_veh['-N1_1'] + self.dict_lane_veh['-S1_1'],
                                self.dict_lane_veh['-E1_2'] + self.dict_lane_veh['-W1_2']
                                ] # lanes_id should be one to one, or may compute wrong
        best_action = np.argmax(self.dict_action_wait_num)
        
        self.calculatePhaseDuration(action)
      
        if action == best_action:
            self.correct_action = 1
        else:
            self.correct_action = -1
            
        if update_reward == True:
            return self.correct_action
        else:
            return None

        
    def calculatePhaseDuration(self, action):
        average_vehicle = self.dict_action_wait_num[action]//2
        self.end_time = self.conn.simulation.getTime()/1000 + average_vehicle * self.conn.lane.getLastStepMeanSpeed('-E1_2')
        self.duration = average_vehicle * self.conn.lane.getLastStepMeanSpeed('-E1_2')
        return average_vehicle * self.conn.lane.getLastStepMeanSpeed('-E1_2')


    def computeNextState(self):
        current_time = self.conn.simulation.getTime()
        if current_time >= self.rs_update_time:
            observation = self.getObservation()
            next_state = np.array(observation, dtype=np.float32)
            return next_state
        else:
            return None


    def computeState(self):
        observation = self.getObservation()
        state = np.array(observation, dtype=np.float32)
        return state


    def getLanesDensity(self):
        lanes_density_with_id = {}
        all_lanes_density = []
        vehicle_size_min_gap = 7.5
        for lane in self.lanes_id:
            lane_density = min(1, self.conn.lane.getLastStepVehicleNumber(lane) / (self.lanes_length[lane] / vehicle_size_min_gap))
            lanes_density_with_id[lane] = lane_density
            all_lanes_density.append(lane_density)
        return all_lanes_density


    def getLanesQueue(self):

        lanes_queue_with_id = {}
        all_lanes_queue = []
        for lane in self.lanes_id:
            lanes_queue = self.conn.lane.getLastStepHaltingNumber(lane) / (self.lanes_length[lane] / max(self.conn.lane.getLastStepLength(lane),1)) # getLastStepLength Returns the mean vehicle length in m for the last time step on the given lane.
            lanes_queue_with_id[lane] = lanes_queue
            all_lanes_queue.append(lanes_queue)
        return all_lanes_queue
    

    def getObservation(self):
        obs = []
        lanes_density = {}
        for i,lane_id in enumerate(self.lanes_id):
            obs.append(self.conn.lane.getLastStepOccupancy(lane_id))
            lanes_density[lane_id] = self.conn.lane.getLastStepOccupancy(lane_id)
        self.lanes_density = lanes_density
        return obs
