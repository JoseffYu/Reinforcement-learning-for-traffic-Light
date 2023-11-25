
import numpy as np
from gym import spaces


class TrafficLight:
    def __init__(
        self,
        tls_id: str,
        simulation_time: int,
        traci
    ):
        self.conn = traci
        self.tls_id = tls_id
        # reward_state_update_time
        self.rs_update_time = 0
        self.green_phase = None
        self.yellow_phase = None
        self.end_time = 0
        self.all_phases = []
        self.lanes_id = []
        self.reward = 0
        self.continue_reward = False
        self.dict_lane_veh = None
        self.simulation_time = simulation_time

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


    def doAction(self, tl_id, action):
        
        new_green_phase_state = self.all_green_phases[action].state
        self.conn.trafficlight.setRedYellowGreenState(tl_id, new_green_phase_state)
        self.green_phase = action
        
        return action
        

    def computeReward(self, action):

        update_reward = True
        
        return self.chooseMinWaitTime(action, update_reward)
    
    
    def chooseMinWaitTime(self,action, update_reward):
        self.dict_lane_veh = {}        
         
        for lane_id in self.lanes_id:
            self.dict_lane_veh[lane_id] = self.getObservation()
        # merge wait_num by actions, position po each lines indicate a  action of traffic light
        dict_action_wait_num = [self.dict_lane_veh['E0_1'] + self.dict_lane_veh['E2_1'],
                                self.dict_lane_veh['-E1_1'] + self.dict_lane_veh['E3_1'],
                                self.dict_lane_veh['-E1_2'] + self.dict_lane_veh['E3_2'],
                                self.dict_lane_veh['E0_2'] + self.dict_lane_veh['E2_1']]
        best_action = np.argmax(dict_action_wait_num)
        if best_action == action:
            self.reward = 1
        else:
            self.reward = -1
            
        if update_reward == True:
            return self.reward
        else:
            return None


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
        vehicle_size_min_gap = 7.5
        for lane in self.lanes_id:
            lane_density = min(1, self.conn.lane.getLastStepVehicleNumber(lane) / (self.lanes_length[lane] / vehicle_size_min_gap))
            lanes_density_with_id[lane] = lane_density
        return lanes_density_with_id


    def getLanesQueue(self):
        """Returns the queue [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        """

        lanes_queue_with_id = {}
        for lane in self.lanes_id:
            lanes_queue = self.conn.lane.getLastStepHaltingNumber(lane) / (self.lanes_length[lane] / max(self.conn.lane.getLastStepLength(lane),1)) # getLastStepLength Returns the mean vehicle length in m for the last time step on the given lane.
            lanes_queue_with_id[lane] = lanes_queue
        return lanes_queue_with_id


    def getObservation(self):
        density = self.getLanesDensity()
        queue = self.getLanesQueue()
        obs = []
        for lane in self.lanes_id:
            obs.append((density[lane]+queue[lane])/2)
        observation = np.array(obs, dtype=np.float32)
        return observation
