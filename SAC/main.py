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
    reward = 10*rewards["lane_centering_reward"]+rewards["action_reward"]-10*rewards["collision_reward"]+10*rewards["on_road_reward"]
    return reward



if __name__ == '__main__':

    times = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
    out_dir = f"train/{times}"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    log_dir = os.path.join(out_dir, 'train.log')
    logger = get_logger(log_dir, name="log", level="info")
    args = parse_args()

    env_name = "racetrack-v0"
    number = 1
    # seed = 0
    env = makeEnv(env_name, args.seed)
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    logger.info("env={}".format(env_name))
    logger.info("state_dim={}".format(state_dim))
    logger.info("action_dim={}".format(action_dim))
    logger.info("max_action={}".format(max_action))

    # device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
    agent = SAC(state_dim, action_dim, max_action, args, logger)
    replay_buffer = ReplayBuffer(state_dim, action_dim, args)
    # Build a tensorboard
    writer = SummaryWriter(log_dir=os.path.join(out_dir, 'SAC_env_{}_number_{}_seed_{}'.format(env_name, number, args.seed)))

<<<<<<< HEAD
    max_train_steps = 5e4  # Maximum number of training steps
    random_steps = 0  # Take the random actions in the beginning for the better exploration
    evaluate_freq = 5e2  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
=======
    max_train_steps = args.total_timesteps  # Maximum number of training steps
    random_steps = args.learning_starts  # Take the random actions in the beginning for the better exploration
    # evaluate_freq = 5e2  # Evaluate the policy every 'evaluate_freq' steps
    # evaluate_num = 0  # Record the number of evaluations
    # evaluate_rewards = []  # Record the rewards during the evaluating
>>>>>>> 9517c5c003171171df2e1e5efc70d9216963f07d
    total_steps = 0  # Record the total steps during the training
    st = time.time()
    total_r = 0
    while total_steps < max_train_steps:
        s,_ = env.reset()
        s = s.flatten()
        episode_steps = 0
        done = False
        truncations =False
        if total_steps > 0: 
            ed = time.time()
            logger.info(f"total_steps: {total_steps}, episode reward: [{total_r}], time_used: {int(ed - st)}")
        total_r = 0
        while not (done or truncations):
            episode_steps += 1
            if total_steps < random_steps:  # Take the random actions in the beginning for the better exploration
                a = env.action_space.sample()
            else:
                a = agent.choose_action(s)
            s_, r, done, truncations, infos = env.step(a)
            r = adjustReward(infos["rewards"])
            if not infos["rewards"]["on_road_reward"]:
                done = True
            # print(infos)
            total_r += r
            s_ = s_.flatten()
            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done or truncations:
                dw = True
            else:
                dw = False
            replay_buffer.store(s, a, r, s_, dw)  # Store the transition
            s = s_

            if total_steps >= random_steps:
                agent.learn(replay_buffer, total_steps=total_steps)

            total_steps += 1
<<<<<<< HEAD
    agent.save("./models/racetrack")
=======
    
>>>>>>> 9517c5c003171171df2e1e5efc70d9216963f07d
