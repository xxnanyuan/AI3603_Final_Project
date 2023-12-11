import gymnasium as gym
import torch
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter
from SAC import SAC
from ReplayBuffer import ReplayBuffer
# from evaluatePolicy import evaluatePolicy
from makeEnv import makeEnv
from utils import get_logger, parse_args
import time
import os


if __name__ == '__main__':

    args = parse_args()
    # log information
    env_name = args.env_name
    # env_name = "intersection-v0"
    # env_name = "racetrack-v0"
    times = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
    out_dir = f"train/{env_name}/{times}"
    if not os.path.exists(f"train"):
        os.mkdir(f"train")
    if not os.path.exists(f"train/{env_name}"):
        os.mkdir(f"train/{env_name}")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    log_dir = os.path.join(out_dir, 'train.log')
    logger = get_logger(log_dir, name="log", level="info")

    number = 1
    # seed = 0
    env = makeEnv(env_name,args.seed)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    # max_action = float(env.action_space.high[0])
    max_action = [1,0.2]

    logger.info("env={}".format(env_name))
    logger.info("state_dim={}".format(state_dim))
    logger.info("action_dim={}".format(action_dim))
    logger.info("max_action={}".format(max_action))

    '''
    1. Input: initial policy parameters theta, Q-function parameters phi_1, phi_2, empty replay buffer D
    2: Set target parameters equal to main parameters phi_targ1 <- phi_1;  phi_targ2 <- phi_2
    (1 and 2 in SAC)
    '''
    agent = SAC(state_dim, action_dim, max_action, args)
    replay_buffer = ReplayBuffer(state_dim, action_dim, args) #empty replay buffer D
    # Build a tensorboard
    writer = SummaryWriter(log_dir=os.path.join(out_dir, 'SAC_env_{}_number_{}_seed_{}'.format(env_name, number, args.seed)))

    max_train_steps = args.total_timesteps  # Maximum number of training steps
    random_steps = args.learning_starts  # Take the random actions in the beginning for the better exploration
    total_steps = 0  # Record the total steps during the training

    st = time.time()
    
    '''4. Observe state s'''
    obs,info = env.reset()
    episode_len = 0
    episode_reward = 0  
    '''3: repeat'''
    while total_steps < max_train_steps:
        '''4. Observe state s'''
        if total_steps < random_steps:  # Take the random actions in the beginning for the better exploration
            action = env.action_space.sample()
        else:
            '''select action'''
            action = np.array(agent.choose_action(obs.flatten()))
            
        next_obs,reward,done,truncations,info = env.step(action)


        reward=0.4*info["rewards"]["high_speed_reward"]+0.2*info["rewards"]["right_lane_reward"]+0.4*(1-info["rewards"]["collision_reward"])+(next_obs[6][5][5]-1)
        reward*=info["rewards"]["on_road_reward"]
        if (np.sum(next_obs[0])<=1):
            reward = 0
        episode_reward += reward
        
        if (done) or (not info["rewards"]["on_road_reward"]):
            dw = True
        else:
            dw = False
            
        replay_buffer.store(obs.flatten(), action, reward, next_obs.flatten(), False)
        episode_len+=1
        
        if total_steps >= random_steps:
            agent.learn(replay_buffer, total_steps=total_steps) 
        
        print(reward)
        if dw or truncations:
            obs, info = env.reset()
            logger.info(f"[episode info] total_steps: {total_steps}, episode lenght: {episode_len}, episode reward: [{episode_reward}], time_used: {int(time.time() - st)}")
            writer.add_scalar("charts/episodic_return", episode_reward, total_steps)
            writer.add_scalar("charts/episodic_length", episode_len, total_steps)     
            if total_steps >= random_steps:   
                writer.add_scalar("charts/alpha", agent.alpha, total_steps)       
            # add tensorboard here
            episode_len=0
            episode_reward=0 
        else:
            obs = np.array(next_obs) 
        
        total_steps += 1
        if total_steps%(1000) == 0:
            logger.info(f"[save info] save model at total_steps: {total_steps}")
            model_path = os.path.join(out_dir, "models/")
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            agent.save(model_path)
            with open (os.path.join(out_dir, "hyperparameters.txt"), 'w') as f: 
                f.write(str(args))