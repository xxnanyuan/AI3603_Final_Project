import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env

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

def act_inference(obs):
    #! Load Your own act_inferece
    action = env.action_space.sample()
    return action 


print(env.action_space,env.observation_space)

if __name__ == '__main__':
    # Create the environment
    
    obs, info = env.reset()

    env = RecordVideo(env, video_folder="racetrack/videos", episode_trigger=lambda e: True)
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