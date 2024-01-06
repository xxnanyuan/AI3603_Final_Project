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



def get_new_obs(obs):
    exist_flag = False
    for i in range(11):
        for j in range(11):
            a = obs[0, i, j]
            if a == 1 and (i, j) != (5, 5): 
                # print(f"({i},{j})")
                exist_flag = True
                i_copy = i
                j_copy = j
    

    obs_onroad = obs[1,3:8,3:8].flatten()
    # obs_onroad = obs[1,4:7,4:7].flatten()
    obs_self = obs[:, 5, 5].flatten()
    if exist_flag:
        obs_next_car = obs[:, i_copy, j_copy].flatten()
    else:
        obs_next_car = np.zeros(obs.shape[0])
    new_obs = np.concatenate((obs_onroad, obs_self, obs_next_car)).flatten()
    return new_obs


if __name__ == '__main__':
    args = parse_args()
    # log information
    env_name = args.env_name
    # env_name = "intersection-v0"
    env_name = "racetrack-v0"
    times = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
    out_dir = f"train/{env_name}_small_feature/{times}"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_dir = os.path.join(out_dir, 'train.log')
    logger = get_logger(log_dir, name="log", level="info")
    

    # init for make env
    num_envs = 1
    env = gym.vector.AsyncVectorEnv(
        [makeEnv(env_name,i+3) for i in range(num_envs)]
    )    
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # set state_dim,action_dim,max_action for four environment
    state_dim = np.prod(env.single_observation_space.shape)
    # change the size of obs
    # state_dim = 72
    state_dim = 41
    # state_dim = 200
    # state_dim = 968
    action_dim = env.single_action_space.shape[0]
    max_action = env.single_action_space.high[:]
    # max_action = [1,0.2]


    logger.info("env={}".format(env_name))
    logger.info("state_dim={}".format(state_dim))
    logger.info("action_dim={}".format(action_dim))
    logger.info("max_action={}".format(max_action))
    # logger.info("the version by cyj, inherited not from fixed version yet the no_final version")
    logger.info("(5*5)但是给，-100的reward")
    # logger.info("reward[i] = -0.2 if out road")

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
    
    # init episode length and reward for log
    episode_len = np.zeros(num_envs)
    episode_reward = np.zeros(num_envs)   
    
    # the sub env will auto reset when terminal
    # and env.reset() will reset all sub env
    # If you want to reset env for some condition, you only can reset all
    # this will cause some problem
    # so we set max episode length for delay the reset action 

    # main loop
    while total_steps < max_train_steps:
        manualResetSignal = np.full(num_envs,False,dtype=bool) 
        if total_steps < random_steps:  # Take the random actions in the beginning for the better exploration
            action = env.action_space.sample()
        else:
            '''select action'''
            # action = np.array([agent.choose_action(obs[i].flatten()) for i in range(num_envs)])
            # change the size of obs
            '''"features": ["presence","on_road", "x", "y", "vx", "vy", "cos_h", "sin_h"]'''

            action = np.array([agent.choose_action(get_new_obs(obs[i])) for i in range(num_envs)])
            # action = np.array([agent.choose_action(obs[i].flatten()) for i in range(num_envs)])
            # print(obs[0].shape)
            # print(get_new_obs(obs[0]).shape)
            # action = np.array([agent.choose_action(obs[i][:][:,4:7,4:7].flatten()) for i in range(num_envs)])
            # action = np.array([agent.choose_action(obs[i][:][:,3:8,3:8].flatten()) for i in range(num_envs)])
        '''
        Step a in the enviroment
        Observe next state s', reward r, and done signal d to indicate whether s' is terminal
        '''    
        next_obs,reward,done,truncations,info = env.step(action)
        
        # now we process for each sub env
        for i in range(num_envs):
            # logger.info(f"new_obs! time_used: {int(time.time() - st)}")
            # new_obs = obs[i].flatten()
            # new_next_obs = next_obs[i].flatten()
            new_obs = get_new_obs(obs[i])
            new_next_obs = get_new_obs(next_obs[i])
            # logger.info(f"new_obs! time_used: {int(time.time() - st)}")
            # change reward
            # reward[i]=0.8*info["rewards"][i]["high_speed_reward"]+0.1*info["rewards"][i]["right_lane_reward"]+0.1-0.1*info["rewards"][i]["collision_reward"]
            if done[i]:
                reward[i]=info["final_info"][i]["rewards"]["lane_centering_reward"]-0.4*info["final_info"][i]["rewards"]["action_reward"]-info["final_info"][i]["rewards"]["collision_reward"]
                reward[i]=(reward[i]-(-1))/(1-(-1))
                reward[i]*=info["final_info"][i]["rewards"]["on_road_reward"]
            #     # print(info["final_info"])

            else:
                reward[i]=info["rewards"][i]["lane_centering_reward"]-0.4*info["rewards"][i]["action_reward"]-info["rewards"][i]["collision_reward"]
                reward[i]=(reward[i]-(-1))/(1-(-1))
                reward[i]*=info["rewards"][i]["on_road_reward"]
            # if (np.sum(next_obs[i][0])<=1):
            #     reward[i] = 0
            #     manualResetSignal[i] = True
                
            if not info["rewards"][i]["on_road_reward"]:
                manualResetSignal[i] = True
                reward[i] -= 100


            # update episode reward
            episode_reward[i] = episode_reward[i] + reward[i]
            if not (done[i] or truncations[i]):
                # if episode don't terminal, the sub env continue
                # store experience in replay buffer 
                # replay_buffer.store(obs[i].flatten(), action[i], reward[i], next_obs[i].flatten(), False)  
                # change the size of obs
                # action = np.array([agent.choose_action(get_new_obs(obs[i][:])) for i in range(num_envs)])
                replay_buffer.store(new_obs, action[i], reward[i], new_next_obs, False)
                # replay_buffer.store(obs[i][:,3:8,3:8].flatten(), action[i], reward[i], next_obs[i][:,3:8,3:8].flatten(), False)
                episode_len[i]+=1
            else:
                # if episode terminal, the env will auto reset
                # the next_obs is obs of the start of next episode
                # the final obs of this episode is store in info["final_observation"][i]
                final_obs = (info["final_observation"][i])
                # store experience in replay buffer 
                # replay_buffer.store(obs[i].flatten(), action[i], reward[i], final_obs.flatten(), done[i])
                # change the size of obs
                replay_buffer.store(new_obs, action[i], reward[i], get_new_obs(final_obs).flatten(), done[i])
                # replay_buffer.store(new_obs, action[i], reward[i], final_obs.flatten(), done[i])
                # replay_buffer.store(obs[i][:,3:8,3:8].flatten(), action[i], reward[i], final_obs[i][:,3:8,3:8].flatten(), False)
                episode_len[i]+=1
                
                # log
                logger.info(f"[episode info] autoReset env_id: {i}, total_steps: {total_steps}, episode lenght: {episode_len[i]}, episode reward: [{episode_reward[i]}], time_used: {int(time.time() - st)}")
                
                # add scalar to tensorboard here
                writer.add_scalar("charts/episodic_return", episode_reward[i], total_steps+i)
                writer.add_scalar("charts/episodic_length", episode_len[i], total_steps+i)  
                if total_steps >= random_steps:   
                    writer.add_scalar("charts/alpha", agent.alpha, total_steps+i)       
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

        if True in manualResetSignal:
            # print(info["rewards"])
            obs, info = env.reset()
            for i in range(num_envs):
                if episode_len[i]!=0:
                    logger.info(f"[episode info] manualReset env_id: {i}, total_steps: {total_steps}, episode lenght: {episode_len[i]}, episode reward: [{episode_reward[i]}], time_used: {int(time.time() - st)}")
                    writer.add_scalar("charts/episodic_return", episode_reward[i], total_steps+i)
                    writer.add_scalar("charts/episodic_length", episode_len[i], total_steps+i)     
                    if total_steps >= random_steps:   
                        writer.add_scalar("charts/alpha", agent.alpha, total_steps+i)       
                    # add tensorboard here
                    episode_len[i]=0
                    episode_reward[i]=0 

        # store model
        if total_steps%(num_envs*200) == 0:
            model_path = os.path.join(out_dir, "models/")
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            agent.save(model_path)
            logger.info(f"save model at step {total_steps}")
            with open (os.path.join(out_dir, "hyperparameters.txt"), 'w') as f: 
                f.write(str(args))