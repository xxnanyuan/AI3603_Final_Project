from typing import Any, Dict, SupportsFloat, Text, Tuple
from gym.core import Env
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np 
import os 
from gymnasium import Wrapper
from models.SAC import SAC

class RaceTrack_v2(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.env = env
    def _is_terminated(self) -> bool:
        return self.env.vehicle.crashed or not self.env.vehicle.on_road
    def step(self, action: Any):
        obs, reward, terminated, truncated, info = super().step(action)
        terminated = self._is_terminated()
        return obs, reward, terminated, truncated, info
    

env = gym.make("racetrack-v0", render_mode="rgb_array")
# env = RecordVideo(env, video_folder=f"videos/racetrack", episode_trigger=lambda e: True)
# env.unwrapped.set_record_video_wrapper(env)
# env.configure({"simulation_frequency": 15})  # Higher FPS for rendering
env.configure({
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 15,
        "features": ["presence","on_road", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
        "grid_step": [5, 5],
        "absolute": False
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": False,
        "lateral": True,
    },

    "duration": 30,  # [s]
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "offroad_terminal":True,
})

env = RaceTrack_v2(env)
env.reset()


def load_act_inference():
    #! Load your own act_inference
    #! act_inference= obs: np.ndarray -> action: np.ndarray
    def get_new_obs(obs):
        exist_flag = False
        for i in range(11):
            for j in range(11):
                a = obs[0, i, j]
                if a == 1 and (i, j) != (5, 5): 
                    # print(f"({i},{j})")
                    exist_flag = True
                    i_copy = i
                    j_copy = j
        
        # obs_onroad = obs[1,4:7,4:7].flatten()
        obs_onroad = obs[1,3:8,3:8].flatten()
        obs_self = obs[:, 5, 5].flatten()
        if exist_flag:
            obs_next_car = obs[:, i_copy, j_copy].flatten()
        else:
            obs_next_car = np.zeros(obs.shape[0])
        new_obs = np.concatenate((obs_onroad, obs_self, obs_next_car)).flatten()
        return new_obs
    agent = SAC(41, 1,  1)
    agent.load("models/racetrack/models/")
    
    return lambda obs: agent.choose_action(get_new_obs(obs),True)



act_inference = load_act_inference()

def eval_racetrack(num_runs,save_path = None):
    eval_results = {}
    list_ep_ret = []
    list_ep_len = []
    for i in range(num_runs):
        ep_ret,ep_len = 0,0
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action = act_inference(obs)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += reward
            ep_len += 1
            # env.render()
        list_ep_ret.append(ep_ret)
        list_ep_len.append(ep_len)
    eval_results['ep_ret'] = np.array(list_ep_ret) 
    eval_results['ep_len'] = np.array(list_ep_len) 

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = save_path + '/eval_results_racetrack.npy' 
        np.save(file_name,eval_results)
    
    for k,v in eval_results.items():
        print(k,f" Mean: {v.mean().round(4)}, Std: {v.std().round(4)}")



            
if __name__ == '__main__':

    eval_racetrack(100,save_path = "eval_files/results")
    env.close()