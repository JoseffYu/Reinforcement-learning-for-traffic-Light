
import numpy as np
from gym import spaces


class TrafficLight:
    def __init__(
        self,
        tl_id: str,
        traci
    ):
        self.conn = traci
        self.tl_id = tl_id
        # reward_state_update_time
        self.rs_update_time = 0
        self.green_phase = None
        self.yellow_phase = None
        self.end_time = 0
        self.all_phases = self.conn.trafficlight.getAllProgramLogics(tl_id)[0].phases
        self.all_green_phases = [phase for phase in self.all_phases if 'g' in phase.state]
        self.lanes_id = list(dict.fromkeys(self.conn.trafficlight.getControlledLanes(self.tl_id)))
        self.lanes_length = {lane_id: self.conn.lane.getLength(lane_id) for lane_id in self.lanes_id}
        self.observation_space = spaces.Box(
            low=np.zeros(len(self.lanes_id), dtype=np.float32),
            high=np.ones(len(self.lanes_id), dtype=np.float32))
        self.action_space = spaces.Discrete(len(self.all_green_phases))
        self.reward = 0
        self.continue_reward = False
        self.dict_lane_veh = None
        
    #TO DO
    def doAction(self, action):
        return 
    
    #TO DO
    def computeReward(self, action):
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