import gymnasium as gym
import torch
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter
from SAC import SAC
from ReplayBuffer import ReplayBuffer
from makeVecEnv import makeEnv
from utils import get_logger, parse_args
import time
import os

if __name__ == '__main__':

    # log information
    env_name = "parking-v0"
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
    args = parse_args()

    # init for make env
    num_envs = 60
    env = gym.vector.AsyncVectorEnv(
        [makeEnv(env_name,i+3) for i in range(num_envs)]
    )    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # set state_dim,action_dim,max_action for four environment
    state_dim = 12
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
    agent = SAC(state_dim, action_dim, max_action, args)
    replay_buffer = ReplayBuffer(state_dim, action_dim, args) 

    # Build a tensorboard
    number = 1
    writer = SummaryWriter(log_dir=os.path.join(out_dir, 'SAC_env_{}_number_{}_seed_{}'.format(env_name, number, args.seed)))

    # pass args
    max_train_steps = args.total_timesteps  # Maximum number of training steps
    random_steps = args.learning_starts  # Take the random actions in the beginning for the better exploration
    total_steps = 0 # Record the total steps during the training

    st = time.time()

    ''' Observe state s'''
    obs,info = env.reset()
    # change the size of obs
    obs = np.array([np.append(obs['achieved_goal'][i],obs['desired_goal'][i]) for i in range(num_envs)])
    
    # init episode length and reward for log
    episode_len = np.zeros(num_envs)
    episode_reward = np.zeros(num_envs)
    
    # we use episode_trace to record all trace of one episode
    # so we can add Hinsight experience to replay buffer
    episode_trace = [[] for _ in range(num_envs)]
        
    while total_steps < max_train_steps:
        if total_steps < random_steps:  # Take the random actions in the beginning for the better exploration
            action = env.action_space.sample()
        else:
            '''select action'''
            action = np.array([agent.choose_action(obs[i].flatten()) for i in range(num_envs)])
        '''
        Step a in the enviroment
        Observe next state s', reward r, and done signal d to indicate whether s' is terminal
        '''    
        next_obs,reward,done,truncations,info = env.step(action)
        # change the size of nwxt_obs
        next_obs = [np.append(next_obs['achieved_goal'][i],next_obs['desired_goal'][i]) for i in range(num_envs)]
        episode_reward = episode_reward + reward
        
        # now we process for each sub env
        for i in range(num_envs):
            if not (done[i] or truncations[i]):
                # if episode don't terminal, the sub env continue
                # store experience in replay buffer 
                replay_buffer.store(obs[i].flatten(), action[i], reward[i], next_obs[i].flatten(), False)  
                # add trace into episode_trace
                episode_trace[i].append([obs[i].flatten()[0:6], action[i], reward[i], next_obs[i].flatten()[0:6],False])
                episode_len[i]+=1
            else:
                # if episode terminal, the env will auto reset
                # the next_obs is obs of the start of next episode
                # the final obs of this episode is store in info["final_observation"][i]
                final_obs = np.append(info["final_observation"][i]['achieved_goal'],info["final_observation"][i]['desired_goal'])
                # store experience in replay buffer 
                replay_buffer.store(obs[i].flatten(), action[i], reward[i], final_obs.flatten(), done[i])
                episode_len[i]+=1
                
                
                # log
                logger.info(f"[episode info] env_id: {i}, total_steps: {total_steps}, episode lenght: {episode_len[i]}, episode reward: [{episode_reward[i]}], time_used: {int(time.time() - st)}, success: {info['final_info'][i]['is_success']}")
                
                # add episode_trace experience to replay buffer
                if not info['final_info'][i]['is_success']:
                    for iterm in episode_trace[i]:
                        new_obs = np.concatenate((iterm[0], obs[i].flatten()[0:6]))
                        new_action = iterm[1]
                        new_reward = -np.power(np.dot(np.abs(iterm[3] - obs[i].flatten()[0:6]),np.array([1, 0.3, 0, 0, 0.02, 0.02]),),0.5)
                        new_next_obs = np.concatenate((iterm[3], obs[i].flatten()[0:6]))
                        new_done = iterm[4]
                        replay_buffer.store(new_obs, new_action, new_reward, new_next_obs, new_done)
                episode_trace[i] = []
                
                # add scalar to tensorboard here
                writer.add_scalar("charts/episodic_return", episode_reward[i], total_steps)
                writer.add_scalar("charts/episodic_length", episode_len[i], total_steps)
                writer.add_scalar("charts/alpha", agent.alpha, total_steps)   
                # init for new episode     
                episode_len[i]=0
                episode_reward[i]=0 
            
            # obs = next_obs
            obs[i] = np.array(next_obs[i]) 
            
            # update agent
            if total_steps >= random_steps:
                agent.learn(replay_buffer, total_steps=total_steps+i) 
        
        # update total_steps          
        total_steps += num_envs
        
        # store model
        if total_steps%num_envs*10 == 0:
            model_path = os.path.join(out_dir, "models/")
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            agent.save(model_path)
            with open (os.path.join(out_dir, "hyperparameters.txt"), 'w') as f: 
                f.write(str(args))