import gymnasium as gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# env = gym.make("Pendulum-v1", render_mode="rgb_array")
env = gym.make("parking-v0", render_mode="rgb_array")
env.configure({
    "observation": {
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,
        "lateral": True,
    },

    "duration": 20,  # [s]
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 5,  # [Hz]
})
env.reset()


model = SAC("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)
model.save("SAC_parking")
vec_env = model.get_env()

del model # remove to demonstrate saving and loading

model = SAC.load("SAC_parking")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()

