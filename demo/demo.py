import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import sys
sys.modules["gym"] = gym
import warnings
warnings.filterwarnings("ignore")

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


model = SAC("MlpPolicy", env, verbose=1,
            tensorboard_log="logs", 
            buffer_size=int(1e3),
            learning_rate=1e-3,
            gamma=0.95, batch_size=24)

model.learn(total_timesteps=500, log_interval=10)
model.save("SAC_highway")


del model # remove to demonstrate saving and loading
env = RecordVideo(env, video_folder="highway/videos", episode_trigger=lambda e: True)
env.unwrapped.set_record_video_wrapper(env)
env.configure({"simulation_frequency": 15})  # Higher FPS for rendering
model = SAC.load("SAC_highway")
obs,_ = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, trunctions, info = env.step(action)
    env.render()