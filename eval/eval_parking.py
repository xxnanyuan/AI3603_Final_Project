import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env

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


print(env.action_space,env.observation_space)

def act_inference(obs):
    #! Load Your own act_inferece
    action = env.action_space.sample()
    return action 


if __name__ == '__main__':
    # Create the environment
    
    obs, info = env.reset()

    env = RecordVideo(env, video_folder="parking/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15})  # Higher FPS for rendering

    for videos in range(1):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action = act_inference(obs)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()