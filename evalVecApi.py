from SAC import SAC
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from utils import get_logger, parse_args
import time
import os
from makeVecEnv import makeEnv

env_name = "racetrack-v0"
times = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
out_dir = f"myeval/{env_name}/{times}"
if not os.path.exists(f"myeval/{env_name}"):
    os.mkdir(f"myeval/{env_name}")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
log_dir = os.path.join(out_dir, 'train.log')
logger = get_logger(log_dir, name="log", level="info")
args = parse_args()

model_time = "2023-11-28 21-30-35"
model_path = f"train/{env_name}/{model_time}/models/"

agent = False
def getAction(env, obs):
    global agent
    if agent == False:
        if env_name == "highway-v0":
            state_dim = np.prod(env.observation_space.shape)
        elif env_name == "parking-v0":
            state_dim = 12
        elif env_name == "intersection-v0":
            state_dim = np.prod(env.observation_space.shape)
        elif env_name == "racetrack-v0":
            state_dim = np.prod(env.observation_space.shape)
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        agent = SAC(state_dim, action_dim, max_action, args, logger)
        agent.load(model_path)
    if env_name == "highway-v0":
        obs = obs.flatten()
    elif env_name == "parking-v0":
        obs = np.append(obs['achieved_goal'],obs['desired_goal']).flatten()
    elif env_name == "intersection-v0":
        obs = obs.flatten()
    elif env_name == "racetrack-v0":
        obs = obs.flatten()
    action = agent.choose_action(obs)
    return action


if __name__ == '__main__':
    # Create the environment
    env = makeEnv(env_name,0)()  
    # env = gym.vector.SyncVectorEnv(
    #    [makeEnv(env_name,args.seed+i) for i in range(1)]
    # )  
    obs, info = env.reset()
    # env = RecordVideo(env, video_folder=f"{out_dir}/videos", episode_trigger=lambda e: True)
    # env.unwrapped.set_record_video_wrapper(env)
    # env.configure({"simulation_frequency": 15})  # Higher FPS for rendering
    success_time = 0
    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        steps = 0
        total_reward = 0
        while not (done or truncated):
            steps += 1
            # Predict
            action = getAction(env,obs)
            # action = env.action_space.sample()
            # action = [0,0]
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # print(reward,obs['achieved_goaldesired_goal'])
            # if not info["rewards"]["on_road_reward"]:
            #     done = True
            # print(info)
            total_reward+=reward
            # if info['is_success']:
            #     success_time+=1
            # if done or truncated:
            #     d = True
            # else:
            #     d = False
            # if d:
            #     print("lallalalalallalala:  ",info["final_observation"].shape)
            # Render
            env.render()
        print(videos,"steps: ", steps, " reward: ", total_reward, "sucess rate: ",success_time/100)
        # print(videos,"steps: ", steps, " reward: ", total_reward)
    env.close()
    