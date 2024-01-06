from SAC import SAC
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from utils import get_logger, parse_args
import time
import os
from makeEnv import makeEnv
from collections import deque

args = parse_args()
env_name = args.env_name
times = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
out_dir = f"myeval/{env_name}/{times}"
if not os.path.exists(f"myeval"):
    os.mkdir(f"myeval")
if not os.path.exists(f"myeval/{env_name}"):
    os.mkdir(f"myeval/{env_name}")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
log_dir = os.path.join(out_dir, 'train.log')
logger = get_logger(log_dir, name="log", level="info")

model_time = args.model_time
model_path = f"train/{env_name}/{model_time}/models/"

agent = False
def getAction(env, obs):
    global agent
    if agent == False:
        if env_name == "highway-v0":
            state_dim = np.prod(env.observation_space.shape)
            state_dim = 31
        elif env_name == "parking-v0":
            state_dim = 12 
        elif env_name == "intersection-v0":
            state_dim = np.prod(env.observation_space.shape)
        elif env_name == "racetrack-v0":
            # state_dim = np.prod(env.observation_space.shape)
            state_dim = 34
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        max_action = [1,0.2]
        agent = SAC(state_dim, action_dim, max_action, args)
        agent.load(model_path)
    if env_name == "highway-v0":
        # change obs size
        # obs = obs[:,3:8,3:8].flatten()
        obs = obs.flatten()
    elif env_name == "parking-v0":
        obs = np.append(obs['achieved_goal'],obs['desired_goal']).flatten()
    elif env_name == "intersection-v0":
        obs = obs.flatten()
    elif env_name == "racetrack-v0":
        # obs = obs[:,4:7,4:7]
        obs = obs.flatten()
        
    action = agent.choose_action(obs,True)
    return action

def changeobs2(obs):
    newobs = [obs[6][5][5],obs[7][5][5]]
    for i in range(3,8):
        newobs.append(obs[1][i][5])
    
    newobs.extend([0,0,0,0,0])
            
    for i in range(3,8):
        if not obs[1][i][5]:
            newobs[i+4]=-1
            newobs.extend([-1,-1,-1,-1])
        flag = False
        for j in range(5,11):
            if i == 5 and j==5:
                continue
            if obs[0][i][j]:
                newobs.extend([obs[2][i][j],obs[3][i][j],obs[4][i][j],obs[5][i][j]])
                newobs[i+4]=1
                flag = True
                break
        if not flag:
            newobs.extend([0,0,0,0])
            
    return np.array(newobs)

def changeobs(obs):
    newobs = [obs[6][5][5],obs[7][5][5]]
    
    for i in range(3,8):
        newobs.append(obs[1][i][5])
            
    for i in range(4,7):
        flag = False
        for j in range(5,11):
            if i == 5 and j==5:
                continue
            if obs[0][i][j]:
                newobs.extend([obs[2][i][j],obs[3][i][j],obs[4][i][j],obs[5][i][j]])
                flag = True
                break
        if not flag:
            newobs.extend([0,0,0,0])

    for i in range(4,7):
        flag = False
        for j in range(4,3,-1):
            if obs[0][i][j]:
                newobs.extend([obs[2][i][j],obs[3][i][j],obs[4][i][j],obs[5][i][j]])
                flag = True
                break
        if not flag:
            newobs.extend([0,0,0,0])
            
    return np.array(newobs)
if __name__ == '__main__':
    # Create the environment
    env = makeEnv(env_name,0)
    # env = gym.vector.SyncVectorEnv(
    #    [makeEnv(env_name,args.seed+i) for i in range(1)]
    # )  
    obs, info = env.reset()
    # env = RecordVideo(env, video_folder=f"{out_dir}/videos", episode_trigger=lambda e: True)
    # env.unwrapped.set_record_video_wrapper(env)
    # env.configure({"simulation_frequency": 15})  # Higher FPS for rendering
    success_time = 0
    total_rewards = []
    epi_steps = []
    for videos in range(100):
        done = truncated = False
        obs, info = env.reset()
        steps = 0
        total_reward = 0
        while not (done or truncated):
            steps += 1
            # Predict
            action = getAction(env,changeobs(obs))
            # action = getAction(env,obs)
            # action = [1,0]
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            if done:
                print(changeobs(last_obs))
                print(reward,info["rewards"])
                success_time+=1
            last_obs=np.copy(obs)
            
            # if info["is_success"]:
            #     success_time+=1
            total_reward+=reward
            # Render
            env.render()
        print(videos,"steps: ", steps, " reward: ", total_reward, "sucess time: ",success_time)
        total_rewards.append(total_reward)
        epi_steps.append(steps)
    print(np.mean(total_rewards),np.mean(epi_steps))

    env.close()
    