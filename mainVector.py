import gymnasium as gym
import torch
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter
from SAC import SAC
from ReplayBuffer import ReplayBuffer
# from evaluatePolicy import evaluatePolicy
from makeVecEnv import makeEnv
from utils import get_logger, parse_args
import time
import os

def adjustReward(rewards):
    reward = rewards["lane_centering_reward"]+1-rewards["collision_reward"]+rewards["on_road_reward"]
    return reward

if __name__ == '__main__':

    # log information
    env_name = "racetrack-v0"
    times = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
    out_dir = f"train/{env_name}/{times}"
    if not os.path.exists(f"train/{env_name}"):
        os.mkdir(f"train/{env_name}")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    log_dir = os.path.join(out_dir, 'train.log')
    logger = get_logger(log_dir, name="log", level="info")
    args = parse_args()

    # init for make env
    number = 1
    num_envs = 12
    env = gym.vector.AsyncVectorEnv(
        [makeEnv(env_name,i+3) for i in range(num_envs)]
    )    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # set state_dim,action_dim,max_action for four environment
    if env_name == "highway-v0":
        state_dim = np.prod(env.single_observation_space.shape)
    elif env_name == "parking-v0":
        state_dim = 12
    elif env_name == "intersection-v0":
        state_dim = np.prod(env.single_observation_space.shape)
    elif env_name == "racetrack-v0":
        state_dim = np.prod(env.single_observation_space.shape)
    action_dim = env.single_action_space.shape[0]
    max_action = float(env.single_action_space.high[0])

    logger.info("env={}".format(env_name))
    logger.info("state_dim={}".format(state_dim))
    logger.info("action_dim={}".format(action_dim))
    logger.info("max_action={}".format(max_action))

    '''
    Input: initial policy parameters theta, Q-function parameters phi_1, phi_2, empty replay buffer D
    Set target parameters equal to main parameters phi_targ1 <- phi_1;  phi_targ2 <- phi_2
    (1 and 2 in SAC)
    '''
    agent = SAC(state_dim, action_dim, max_action, args, logger)
    replay_buffer = ReplayBuffer(state_dim, action_dim, args) #empty replay buffer D
    # Build a tensorboard
    writer = SummaryWriter(log_dir=os.path.join(out_dir, 'SAC_env_{}_number_{}_seed_{}'.format(env_name, number, args.seed)))

    max_train_steps = args.total_timesteps  # Maximum number of training steps
    random_steps = args.learning_starts  # Take the random actions in the beginning for the better exploration
    total_steps = 0 # Record the total steps during the training

    st = time.time()

    ''' Observe state s'''
    s,info = env.reset()
    
    episode_len = [0 for i in range(num_envs)]
    episode_reward = [0 for i in range(num_envs)]
    max_episode_len = 1
    while total_steps < max_train_steps:
        # adjust s for input
        if env_name == "highway-v0":
            s = [s[i].flatten() for i in range(num_envs)]
        elif env_name == "parking-v0":
            s = [np.append(s['achieved_goal'][i],s['desired_goal'][i]).flatten() for i in range(num_envs)]
        elif env_name == "intersection-v0":
            s = [s[i].flatten() for i in range(num_envs)]
        elif env_name == "racetrack-v0":
            s = [s[i].flatten() for i in range(num_envs)]
        '''initialize the signals'''
        done = [False for _ in range(num_envs)]
        truncations = [False for _ in range(num_envs)]
        onWay = [True for _ in range(num_envs)]
        d = [False for _ in range(num_envs)]
        if total_steps < random_steps:  # Take the random actions in the beginning for the better exploration
            a = env.action_space.sample()
        else:
            '''select action'''
            a = [agent.choose_action(s[i]) for i in range(num_envs)]
        '''
        Step a in the enviroment
        Observe next state s', reward r, and done signal d to indicate whether s' is terminal
        '''    
        s_, r, done, truncations, info = env.step(a)
        episode_reward += r

        if env_name == "highway-v0":
            for i in range(num_envs):
                if not info["rewards"][i]["on_road_reward"]:
                    onWay[i] = False
            s_ = [s_[i].flatten() for i in range(num_envs)]
        elif env_name == "parking-v0":
            s_ = [np.append(s_['achieved_goal'][i],s_['desired_goal'][i]).flatten() for i in range(num_envs)]
        elif env_name == "intersection-v0":
            for i in range(num_envs):
                if not info["rewards"][i]["on_road_reward"]:
                    onWay[i] = False
            s_ = [s_[i].flatten() for i in range(num_envs)]
        elif env_name == "racetrack-v0":
            for i in range(num_envs):
                if not info["rewards"][i]["on_road_reward"]:
                    onWay[i] = False
            s_ = [s_[i].flatten() for i in range(num_envs)]
        
        '''observe signal d'''
        for i in range(num_envs):
            if done[i] or truncations[i]:
                d[i] = True
            else:
                d[i] = False
        
            '''Store (s,a,r,s', d) in replay buffer D'''
            if d[i]:
                replay_buffer.store(s[i], a[i], r[i], info["final_observation"][i].flatten(), d[i])
            else:
                replay_buffer.store(s[i], a[i], r[i], s_[i], d[i])  
            
            s[i] = np.array(s_[i]) 
            '''
            if it's time to update then do the learn process, 
            10. for j in range(however many updates) 
            the following 11~15 are all in func "learn". 
            '''
            if total_steps >= random_steps:
                agent.learn(replay_buffer, total_steps=total_steps+i)
            
            episode_len[i]+=1 
                
            if d[i]:
                if env_name == "parking-v0":
                    logger.info(f"[episode info] env_id: {i}, total_steps: {total_steps}, episode lenght: {episode_len[i]}, episode reward: [{episode_reward[i]}], time_used: {int(time.time() - st)}, success: {info['is_success'][i]}")
                    episode_len[i]=0
                    episode_reward[i]=0
                else:
                    logger.info(f"[episode info] env_id: {i}, total_steps: {total_steps}, episode lenght: {episode_len[i]}, episode reward: [{episode_reward[i]}], time_used: {int(time.time() - st)}")
                    episode_len[i]=0
                    episode_reward[i]=0

        # if env_name != "parking-v0":
        #     logger.info(f"[step info] total_steps: {total_steps}, step reward: {np.mean(r)}, time_used: {int(time.time() - st)}")
        
        max_episode_len = max(min(episode_len),max_episode_len)
        total_steps += num_envs
        if False in onWay and np.mean(episode_len)>=max_episode_len:
            s, _ = env.reset()
            for i in range(num_envs):
                if env_name == "parking-v0":
                    logger.info(f"[episode info] env_id: {i}, total_steps: {total_steps}, episode lenght: {episode_len[i]}, episode reward: [{episode_reward[i]}], time_used: {int(time.time() - st)}, success: {info['is_success'][i]}")
                    episode_len[i]=0
                    episode_reward[i]=0
                else:
                    logger.info(f"[episode info] env_id: {i}, total_steps: {total_steps}, episode lenght: {episode_len[i]}, episode reward: [{episode_reward[i]}], time_used: {int(time.time() - st)}")
                    episode_len[i]=0
                    episode_reward[i]=0
                    
        if total_steps%(num_envs*500) == 0:
            model_path = os.path.join(out_dir, "models/")
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            agent.save(model_path)
            with open (os.path.join(out_dir, "hyperparameters.txt"), 'w') as f: 
                f.write(str(args))
           
            