import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np 
import os 
from models.SAC import SAC
env = gym.make("highway-v0",render_mode="rgb_array")
# env = RecordVideo(env, video_folder=f"videos/highway", episode_trigger=lambda e: True)
# env.unwrapped.set_record_video_wrapper(env)
# env.configure({"simulation_frequency": 15})  # Higher FPS for rendering
env.configure({
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 15,
        "features": ["presence", "on_road","x", "y", "vx", "vy", "cos_h", "sin_h"],
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

    "duration": 40,  # [s]
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "normalize_reward": False,
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
    agent = SAC(31, 2, [1,0.2])
    agent.load("models/highway/models/")
    def changeobs(obs):
        newobs = [obs[6][5][5],obs[7][5][5]]
        
        for i in range(3,8):
            newobs.append(obs[1][i][5])
                
        for i in range(4,7):
            flag = False
            for j in range(5,11):
                if i == 5 and j==5:
                    continue
                if obs[0][i][j]:
                    newobs.extend([obs[2][i][j],obs[3][i][j],obs[4][i][j],obs[5][i][j]])
                    flag = True
                    break
            if not flag:
                newobs.extend([0,0,0,0])

        for i in range(4,7):
            flag = False
            for j in range(4,2,-1):
                if obs[0][i][j]:
                    newobs.extend([obs[2][i][j],obs[3][i][j],obs[4][i][j],obs[5][i][j]])
                    flag = True
                    break
            if not flag:
                newobs.extend([0,0,0,0])
                
        return np.array(newobs)
    return lambda obs: agent.choose_action(changeobs(obs),True)



act_inference = load_act_inference()

def eval_highway(num_runs,save_path = None):
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
        # print(ep_ret,ep_len)
    eval_results['ep_ret'] = np.array(list_ep_ret) 
    eval_results['ep_len'] = np.array(list_ep_len) 

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = save_path + '/eval_results_highway.npy' 
        np.save(file_name,eval_results)
    
    for k,v in eval_results.items():
        print(k,f" Mean: {v.mean().round(4)}, Std: {v.std().round(4)}")



            



if __name__ == '__main__':

    eval_highway(100,save_path = "eval_files/results")
    env.close()