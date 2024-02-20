import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np 
import os 
from models.SAC import SAC

env = gym.make("intersection-v0", render_mode="rgb_array")
# env = RecordVideo(env, video_folder=f"videos/intersection/", episode_trigger=lambda e: True)
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
        "longitudinal": True,
        "lateral": True,
    },

    "duration": 13,  # [s]
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "normalize_reward": True,
    "offroad_terminal": True,
})

env.reset()

def load_act_inference():
    #! Load your own act_inference
    #! act_inference= obs: np.ndarray -> action: np.ndarray
    """
    Example:

    act_inference = model.forward 
    return act_inference

    """
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    max_action = [1,0]
    agent = SAC(state_dim, action_dim, max_action)
    agent.load("models/intersection/models/")
   
    action = lambda obs: agent.choose_action(obs.flatten(), True)
    
    return action


act_inference = load_act_inference()

def eval_intersection(num_runs,save_path = None):
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
            # env.render()
            ep_ret += reward
            ep_len += 1
        list_ep_ret.append(ep_ret)
        list_ep_len.append(ep_len)
    eval_results['ep_ret'] = np.array(list_ep_ret) 
    eval_results['ep_len'] = np.array(list_ep_len) 

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = save_path + '/eval_results_intersection.npy' 
        np.save(file_name,eval_results)
    
    for k,v in eval_results.items():
        print(k,f" Mean: {v.mean().round(4)}, Std: {v.std().round(4)}")



            



if __name__ == '__main__':

    eval_intersection(100,save_path = "eval_files/results")
    env.close()