
import numpy as np
from gym import spaces


class TrafficLight:
    def __init__(
        self,
        tls_id: str,
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

        self.observation_space = spaces.Box(
            low=np.zeros(len(self.lanes_id), dtype=np.float32),
            high=np.ones(len(self.lanes_id), dtype=np.float32))
        self.action_space = spaces.Discrete(len(self.all_green_phases))
        
    #TO DO
    def doAction(self, tl_id, action):
        
        num_tl = self.tls_id.index(tl_id)
        
        if action > len(self.all_green_phases):
            raise IndexError
        
        #to do
        new_green_phase = self.all_green_phases[action]
        self.conn.trafficlight.setRedYellowGreenState(self.tl_id, new_green_phase)
        
        return action

    
    #TO DO
    def computeReward(self, action):
        update_reward = False
        current_time = self.conn.simulation.getTime()
        if current_time >= self.rs_update_time:
            # set rs_update_time unreachable
            self.rs_update_time = self.simulation_time + self.delta_rs_update_time
            update_reward = True
        
        return self.choose_min_wait_time(action)
    
    
    def choose_min_wait_time(self,action):
        self.dict_lane_veh = {}
         
         
        for lane_id in self.lanes_id:
            self.dict_lane_veh[lane_id] = self.sumo.lane.getLastStepHaltingNumber(lane_id)
            # merge wait_num by actions
            dict_action_wait_num = [self.dict_lane_veh['E0_1'] + self.dict_lane_veh['E2_1'],
                                    self.dict_lane_veh['-E1_1'] + self.dict_lane_veh['E3_1'],
                                    self.dict_lane_veh['E0_2'] + self.dict_lane_veh['E2_1'],
                                    self.dict_lane_veh['-E1_2'] + self.dict_lane_veh['E3_2']]
            best_action = np.argmax(dict_action_wait_num)
        if best_action == action:
            self.reward += 1
        else:
            self.reward -= 1

        return self.reward


    def computeNextState(self):
        current_time = self.conn.simulation.getTime()
        if current_time >= self.rs_update_time:
            density = self.get_lanes_density()
            next_state = np.array(density, dtype=np.float32)
            return next_state
        else:
            return None


    def computeState(self):
        density = self.get_lanes_density()
        state = np.array(density, dtype=np.float32)
        return state


    def getLanesDensity(self):
        vehicle_size_min_gap = 7.5
        return [min(1, self.conn.lane.getLastStepVehicleNumber(lane_id) / (self.lanes_length[lane_id] / vehicle_size_min_gap))
                for lane_id in self.lanes_id]
