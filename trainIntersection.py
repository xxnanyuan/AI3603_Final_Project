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
import math
if __name__ == '__main__':


    # def get_angle(sin_value, cos_value):
    # # 使用 math.atan2 计算角度（以弧度为单位）
    #     angle_rad = math.atan2(sin_value, cos_value)
        
    #     # 将角度从弧度转换为度数
    #     angle_deg = math.degrees(angle_rad)
        
    #     # 将角度转换为 -180 到 180 度的范围
    #     angle_deg = (angle_deg + 180) % 360 - 180
        
    #     return angle_deg
    args = parse_args()
    # log information
    # env_name = args.env_name
    env_name = "intersection-v0"
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
    

    # init for make env
    # num_envs = 24
    # env = gym.vector.AsyncVectorEnv(
    #     [makeEnv(env_name,i+3) for i in range(num_envs)]
    # )    
    env = makeEnv(env_name, 0)()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # set state_dim,action_dim,max_action for four environment
    state_dim = 968
    # change the size of obs
    # state_dim = 200
    action_dim = 2
    # max_action = env.action_space.high[:]
    max_action = [1,0]

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
    # next_obs = obs
    # init episode length and reward for log
    episode_len = 0
    episode_reward = 0
    
    # the sub env will auto reset when terminal
    # and env.reset() will reset all sub env
    max_episode_len = 1
    # main loop
    while total_steps < max_train_steps:
        if total_steps < 100:  # Take the random actions in the beginning for the better exploration
            # action = env.action_space.sample()
            action = [0, 0]
        else:
            '''select action'''
            action = agent.choose_action(obs.flatten())
            action[1] = 0
            # angle = get_angle(obs[7][5][5],obs[6][5][5])
            # if angle >= -89 and angle <= 0:
            #     # action = agent.choose_action(obs.flatten())
            #     while action[1] >= 0:
            #         action = agent.choose_action(obs.flatten())
            # elif angle >= -180 and angle <= -91:
            #     # action = agent.choose_action(obs.flatten())
            #     while action[1] <= 0:
            #         action = agent.choose_action(obs.flatten())
            # elif angle >= -91 and angle <= -89:
            #     action[1] = 0
            # if np.sum(obs[1][5] == 2):
            #     action[1] = 0
            
            # action = agent.choose_action(obs.flatten())
            # change the size of obs
            # action = np.array([agent.choose_action(obs[i][:][:,3:8,3:8].flatten()) for i in range(num_envs)])
        '''
        Step a in the enviroment
        Observe next state s', reward r, and done signal d to indicate whether s' is terminal
        '''
        # if total_steps > 1000:
        #     env.render()    
        next_obs,reward,done,truncations,info = env.step(action)

        # if not info["rewards"]["on_road_reward"]:
        #     reward -= 10
        #     done = True

        # if np.sum(obs[1][4:7,4:7]) >= 8:
        #     reward += 100
        # # if info["rewards"]["arrived_reward"] == 0:
        # #     reward = -1
        # if (info["rewards"]["arrived_reward"] > 0) and info["rewards"]["on_road_reward"] and episode_len >= 7:
        #     reward += 10000
        #     done = True
        # if (info["rewards"]["arrived_reward"] > 0) and episode_len <= 5:
        #     reward -= 10
        #     done = True
        # now we process for each sub env
        # for i in range(num_envs):
        # if done:
            # change reward
            # reward=0.45*info["final_info"]["rewards"]["high_speed_reward"]+0.1*info["final_info"]["rewards"]["right_lane_reward"]+0.45*(1-info["final_info"]["rewards"]["collision_reward"])
            # reward *=info["final_info"]["rewards"]["on_road_reward"]
        # else:        
            # change reward
            # reward=0.45*info["rewards"]["high_speed_reward"]+0.1*info["rewards"]["right_lane_reward"]+0.45*(1-info["rewards"]["collision_reward"])-2*abs(action[1])
            # reward*=info["rewards"]["on_road_reward"]
            # if (np.sum(next_obs[i][0])<=1):
                # reward[i] = 0

            # update episode reward
        episode_reward = episode_reward + reward
        if not (done or truncations):
            # if episode don't terminal, the sub env continue
            # store experience in replay buffer 
            replay_buffer.store(obs.flatten(), action, reward, next_obs.flatten(), False)
            # change the size of obs
            # replay_buffer.store(obs[i][:,3:8,3:8].flatten(), action[i], reward[i], next_obs[i][:,3:8,3:8].flatten(), False or (not info["rewards"][i]["on_road_reward"]))
            episode_len += 1

            obs = next_obs
        else:
            # if episode terminal, the env will auto reset
            # the next_obs is obs of the start of next episode
            # the final obs of this episode is store in info["final_observation"][i]
            if truncations:
            # store experience in replay buffer 
                replay_buffer.store(obs.flatten(), action, reward, next_obs.flatten(), False)
            else:
                replay_buffer.store(obs.flatten(), action, reward, next_obs.flatten(), True)
            # change the size of obs
            # replay_buffer.store(obs[i][:,3:8,3:8].flatten(), action[i], reward[i], final_obs[:,3:8,3:8].flatten(), done[i] or (not info["rewards"][i]["on_road_reward"]))
            episode_len+=1
            
            # log
            logger.info(f"[episode info] env_id: {0}, total_steps: {total_steps}, episode length: {episode_len}, episode reward: [{episode_reward}], time_used: {int(time.time() - st)}")
            
            # add scalar to tensorboard here
            writer.add_scalar("charts/episodic_return", episode_reward, total_steps)
            writer.add_scalar("charts/episodic_length", episode_len, total_steps)  
            if total_steps >= random_steps:   
                writer.add_scalar("charts/alpha", agent.alpha, total_steps)       
            # init for new episode  
            episode_len=0
            episode_reward=0 
            obs, info = env.reset()
            # next_obs = obs
        
        # obs = next_obs
        # obs = next_obs
        
        # update agent
        if total_steps >= 10:
            agent.learn(replay_buffer, total_steps=total_steps) 
            
        # max_episode_len = max(max_episode_len,episode_len)
        # if max_episode_len<=7:
            #  if the car isn't work as you like(store in onTheWay), all environment will reset here  
        # print(np.mean(reward[done==False]),np.min(reward[done==False]),(np.min(reward[done==False])<=0))
        # if (np.min(reward[done==False])<=0):
        #     obs, info = env.reset()
        #     for i in range(num_envs):
        #         if episode_len[i]!=0:
        #             logger.info(f"[episode info] manualReset env_id: {i}, total_steps: {total_steps}, episode lenght: {episode_len[i]}, episode reward: [{episode_reward[i]}], time_used: {int(time.time() - st)}")
        #             writer.add_scalar("charts/episodic_return", episode_reward[i], total_steps+i)
        #             writer.add_scalar("charts/episodic_length", episode_len[i], total_steps+i)     
        #             if total_steps >= random_steps:   
        #                 writer.add_scalar("charts/alpha", agent.alpha, total_steps+i)       
        #             # add tensorboard here
        #             episode_len[i]=0
        #             episode_reward[i]=0 

        # update total_steps
        total_steps += 1      
        # if agent.alpha<0.1:
        #     agent.adaptive_alpha = False
        #     agent.alpha = 0.1
        
        # if max(episode_reward)>35:
        #     agent.adaptive_alpha = True

        # store model
        if total_steps%(500) == 0:
            logger.info(f"[save info] save model at total_steps: {total_steps}")
            model_path = os.path.join(out_dir, "models/")
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            agent.save(model_path)
            with open (os.path.join(out_dir, "hyperparameters.txt"), 'w') as f: 
                f.write(str(args))