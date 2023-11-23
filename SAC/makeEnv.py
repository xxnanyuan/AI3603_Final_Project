import gymnasium as gym

def makeEnv(env_name, seed):
    env = gym.make(env_name, render_mode="rgb_array")
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
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
