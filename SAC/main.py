import gymnasium as gym
import torch
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter
from SAC import SAC
from ReplayBuffer import ReplayBuffer
from evaluatePolicy import evaluatePolicy
from makeEnv import makeEnv

if __name__ == '__main__':
    env_name = "racetrack-v0"
    number = 1
    seed = 0
    env = makeEnv(env_name, seed)
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print("env={}".format(env_name))
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_action={}".format(max_action))

    agent = SAC(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/SAC/SAC_env_{}_number_{}_seed_{}'.format(env_name, number, seed))

    max_train_steps = 5e4  # Maximum number of training steps
    random_steps = 1e4  # Take the random actions in the beginning for the better exploration
    evaluate_freq = 5e2  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    while total_steps < max_train_steps:
        s,_ = env.reset()
        s = s.flatten()
        episode_steps = 0
        done = False
        truncations =False
        while not (done or truncations):
            episode_steps += 1
            if total_steps < random_steps:  # Take the random actions in the beginning for the better exploration
                a = env.action_space.sample()
            else:
                a = agent.choose_action(s)
            s_, r, done, truncations, infos = env.step(a)
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
                agent.learn(replay_buffer)

            total_steps += 1