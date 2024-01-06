import gymnasium as gym

def makeEnv(env_name,seed):
    def thunk():
        env = gym.make(env_name, render_mode="rgb_array")
        if env_name == "highway-v0":
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
        elif env_name == "intersection-v0":
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
        elif env_name == "parking-v0":
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
        elif env_name == "racetrack-v0":
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
                "offroad_terminal":True,
            })
        env.reset()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk
