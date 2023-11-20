import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN, DDPG, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env


TRAIN = True

if __name__ == '__main__':
    n_cpu = 6
    batch_size = 64
    env = make_vec_env("racetrack-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    env = gym.make("highway-v0", render_mode="rgb_array")

    # env.configure({
    #     "observation": {
    #         "type": "OccupancyGrid",
    #         "vehicles_count": 15,
    #         "features": ["presence", "on_road","x", "y", "vx", "vy", "cos_h", "sin_h"],
    #         "features_range": {
    #             "x": [-100, 100],
    #             "y": [-100, 100],
    #             "vx": [-20, 20],
    #             "vy": [-20, 20]
    #         },
    #         "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
    #         "grid_step": [5, 5],
    #         "absolute": False
    #     },
    #     "action": {
    #         "type": "ContinuousAction",
    #         "longitudinal": True,
    #         "lateral": True,
    #     },

    #     "duration": 40,  # [s]
    #     "simulation_frequency": 15,  # [Hz]
    #     "policy_frequency": 1,  # [Hz]
    # })
    env.reset()

    model = PPO("MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                n_steps=batch_size * 12 // n_cpu,
                batch_size=batch_size,
                n_epochs=10,
                learning_rate=5e-4,
                gamma=0.9,
                verbose=2,
                tensorboard_log="racetrack_ppo/")
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(2e3))
        model.save("racetrack_ppo/model")
        del model

    # Run the algorithm
    model = PPO.load("racetrack_ppo/model", env=env)

    env = gym.make("highway-v0", render_mode="rgb_array")

    # env.configure({
    #     "observation": {
    #         "type": "OccupancyGrid",
    #         "vehicles_count": 15,
    #         "features": ["presence", "on_road","x", "y", "vx", "vy", "cos_h", "sin_h"],
    #         "features_range": {
    #             "x": [-100, 100],
    #             "y": [-100, 100],
    #             "vx": [-20, 20],
    #             "vy": [-20, 20]
    #         },
    #         "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
    #         "grid_step": [5, 5],
    #         "absolute": False
    #     },
    #     "action": {
    #         "type": "ContinuousAction",
    #         "longitudinal": True,
    #         "lateral": True,
    #     },

    #     "duration": 40,  # [s]
    #     "simulation_frequency": 15,  # [Hz]
    #     "policy_frequency": 1,  # [Hz]
    # })
    env.reset()
    # env = RecordVideo(env, video_folder="racetrack_ppo/videos", episode_trigger=lambda e: True)
    # env.unwrapped.set_record_video_wrapper(env)

    for video in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # print(obs[1])
            # Render
            env.render()
    env.close()