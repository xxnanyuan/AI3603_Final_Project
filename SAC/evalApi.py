from SAC import SAC
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env
from utils import get_logger, parse_args
import time
import os

times = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
out_dir = f"eval/{times}"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
log_dir = os.path.join(out_dir, 'train.log')
logger = get_logger(log_dir, name="log", level="info")
args = parse_args()

model_dir = "train/2023-11-24 21-00-19/models_racetrack/"

agent = False
def getAction(env, obs):
    global agent
    if agent == False:
        state_dim = np.prod(env.observation_space.shape)
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        agent = SAC(state_dim, action_dim, max_action, args, logger)
        agent.load(model_dir)
    obs = obs.flatten()
    action = agent.choose_action(obs)
    return action


env = gym.make("racetrack-v0", render_mode="rgb_array")
env = RecordVideo(env, video_folder= out_dir + "/videos", episode_trigger=lambda e: True)
env.unwrapped.set_record_video_wrapper(env)

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
})
env.reset()

print(env.action_space,env.observation_space)

if __name__ == '__main__':
    # Create the environment
    
    obs, info = env.reset()

    # env = RecordVideo(env, video_folder="racetrack/videos", episode_trigger=lambda e: True)
    # env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15})  # Higher FPS for rendering

    for videos in range(1):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action = getAction(env, obs)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()
    