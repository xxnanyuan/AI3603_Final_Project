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

def adjustReward(rewards):
    reward = rewards["lane_centering_reward"]+1-rewards["collision_reward"]+rewards["on_road_reward"]
    return reward

if __name__ == '__main__':

    env_name = "highway-v0"
    times = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
    out_dir = f"train/{env_name}/{times}"
    if not os.path.exists(f"train/{env_name}"):
        os.mkdir(f"train/{env_name}")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    log_dir = os.path.join(out_dir, 'train.log')
    logger = get_logger(log_dir, name="log", level="info")
    args = parse_args()

    number = 1
    # seed = 0
    env = makeEnv(env_name)
    # Set random seed
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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

    logger.info("env={}".format(env_name))
    logger.info("state_dim={}".format(state_dim))
    logger.info("action_dim={}".format(action_dim))
    logger.info("max_action={}".format(max_action))

    '''
    1. Input: initial policy parameters theta, Q-function parameters phi_1, phi_2, empty replay buffer D
    2: Set target parameters equal to main parameters phi_targ1 <- phi_1;  phi_targ2 <- phi_2
    (1 and 2 in SAC)
    '''
    agent = SAC(state_dim, action_dim, max_action, args, logger)
    replay_buffer = ReplayBuffer(state_dim, action_dim, args) #empty replay buffer D
    # Build a tensorboard
    writer = SummaryWriter(log_dir=os.path.join(out_dir, 'SAC_env_{}_number_{}_seed_{}'.format(env_name, number, args.seed)))

    max_train_steps = args.total_timesteps  # Maximum number of training steps
    random_steps = args.learning_starts  # Take the random actions in the beginning for the better exploration
    total_steps = 0  # Record the total steps during the training
    last_step = 0
    st = time.time()
    total_r = 0
    '''3: repeat'''
    while total_steps < max_train_steps:
        '''4. Observe state s'''
        s,infos = env.reset()
        # s = s[:,3:8,3:8]
        if env_name == "highway-v0":
            s = s.flatten()
        elif env_name == "parking-v0":
            s = np.append(s['achieved_goal'],s['desired_goal']).flatten()
        elif env_name == "intersection-v0":
            s = s.flatten()
        elif env_name == "racetrack-v0":
            s = s.flatten()

        '''initialize the signals'''
        done = False
        truncations =False
        if done or truncations:
            d = True
        else:
            d = False

        total_r = 0

        while not d:
            if total_steps < random_steps:  # Take the random actions in the beginning for the better exploration
                a = env.action_space.sample()
            else:
                '''4. select action'''
                a = agent.choose_action(s)
            '''
            5.step a in the enviroment
            6. Observe next state s', reward r, and done signal d to indicate whether s' is terminal
            '''    
            s_, r, done, truncations, infos = env.step(a)
            total_r += r
            # if not (1 in s_[1][3:8,3:8]):
            #     done = True
            if env_name == "highway-v0":
                if not infos["rewards"]["on_road_reward"]:
                    done = True
                s_ = s_.flatten()
            elif env_name == "parking-v0":
                s_ = np.append(s_['achieved_goal'],s_['desired_goal']).flatten()
            elif env_name == "intersection-v0":
                if not infos["rewards"]["on_road_reward"]:
                    done = True
                s_ = s_.flatten()
            elif env_name == "racetrack-v0":
                if not infos["rewards"]["on_road_reward"]:
                    done = True
                s_ = s_.flatten()
            
            '''6. observe signal d'''
            if done or truncations:
                d = True
            else:
                d = False
            
            '''7. Store (s,a,r,s', d) in replay buffer D'''
            replay_buffer.store(s, a, r, s_, d)
            s = s_
            
            '''
            9. if it's time to update then do the learn process, 
            10. for j in range(however many updates) 
            the following 11~15 are all in func "learn". 
            '''
            if total_steps >= random_steps:
                agent.learn(replay_buffer, total_steps=total_steps)

            total_steps += 1

            '''8. If s'is terminal,then d is true, we jump out of the "while" and reset environment state.'''
            if total_steps%3000 == 0:
                model_path = os.path.join(out_dir, "models/")
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                agent.save(model_path)
                with open (os.path.join(out_dir, "hyperparameters.txt"), 'w') as f: 
                    f.write(str(args))
                    
        if total_steps > 0: 
            ed = time.time()
            if env_name == "parking-v0":
                logger.info(f"total_steps: {total_steps}, episode reward: [{total_r}], episode length: [{(total_steps-last_step)}], mean step reward: [{total_r/(total_steps-last_step)}], time_used: {int(ed - st)}, success: {infos['is_success']}")
            else:
                logger.info(f"total_steps: {total_steps}, episode reward: [{total_r}], episode length: [{(total_steps-last_step)}], mean step reward: [{total_r/(total_steps-last_step)}], time_used: {int(ed - st)}")
            last_step = total_steps
