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

agent = False
def getAction(env, obs):
    global agent
    if agent == False:
        state_dim = 72
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        agent = SAC(state_dim, action_dim, max_action, args, logger)
        agent.load(f"./models/racetrack")
    obs = obs[:,3:8,3:8]
    obs = obs.flatten()
    action = agent.choose_action(obs)
    return action


env = gym.make("racetrack-v0", render_mode="rgb_array")

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
    # env.configure({"simulation_frequency": 15})  # Higher FPS for rendering

    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        steps = 0
        total_reward = 0
        while not (done or truncated):
            steps += 1
            # Predict
            action = getAction(env, obs)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # if not info["rewards"]["on_road_reward"]:
            #     done = True
            # print(obs[2])
            total_reward+=reward
            # Render
            env.render()
        print("steps: ", steps, " reward: ", total_reward)
    env.close()
    