# -*- coding:utf-8 -*-
import argparse
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import os
import logging as py_logging
import argparse
from distutils.util import strtobool

_log_level = {
    None: py_logging.NOTSET,
    "debug": py_logging.DEBUG,
    "info": py_logging.INFO,
    "warning": py_logging.WARNING,
    "error": py_logging.ERROR,
    "critical": py_logging.CRITICAL
}

def get_logger(
    log_file_path=None,
    name="default_log",
    level=None
):
    directory = os.path.dirname(log_file_path)
    if os.path.isdir(directory) and not os.path.exists(directory):
        os.makedirs(directory)

    root_logger = py_logging.getLogger(name)
    handlers = root_logger.handlers

    def _check_file_handler(logger, filepath):
        for handler in logger.handlers:
            if isinstance(handler, py_logging.FileHandler):
                handler.baseFilename
                return handler.baseFilename == os.path.abspath(filepath)
        return False

    if (log_file_path is not None and not
            _check_file_handler(root_logger, log_file_path)):
        log_formatter = py_logging.Formatter(
            "%(asctime)s [%(levelname)-5.5s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")
        file_handler = py_logging.FileHandler(log_file_path)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    if any([type(h) == py_logging.StreamHandler for h in handlers]):
        return root_logger
    level_format = "\x1b[36m[%(levelname)-5.5s]\x1b[0m"
    log_formatter = py_logging.Formatter(f"{level_format} %(message)s")
    console_handler = py_logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(_log_level[level])
    return root_logger

def parse_args():
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=0.001,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=30000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=4096,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=0.6,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.1,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.1,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=1000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=100,
        help="the frequency of training")
    args = parser.parse_args()
    args.env_id = "intersection-v0"
    return args

def make_env(env_id, seed):
    """construct the gym environment"""
    env = gym.make(env_id, render_mode="rgb_array")
    env.configure({
        "observation": {
            "type": "OccupancyGrid",
            "vehicles_count": 15,
            "features": ["presence","on_road", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
            "grid_step": [5, 5],
            "absolute": False
        },
        "action": {
            "type": "DiscreteAction",
            "longitudinal": True,
            "lateral": True,
            "actions_per_axis":5,
        },

        "duration": 13,  # [s]
        "simulation_frequency": 15,  # [Hz]
        "policy_frequency": 1,  # [Hz]
    })
    env.reset()
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

class QNetwork(nn.Module):
    """comments: """
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """comments: """
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)

    """parse the arguments"""
    args = parse_args()
    env_name = args.env_id
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

    envs = make_env(args.env_id, args.seed)

    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    logger.info("env={}".format(env_name))
    
    """we utilize tensorboard yo log the training process"""
    number = 1
    writer = SummaryWriter(log_dir=os.path.join(out_dir, 'DQN_env_{}_number_{}_seed_{}'.format(env_name, number, args.seed)))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    """comments: """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    """comments: """
    
    
    # print(envs.observation_space)
    """comments: """
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    """comments: """
    rb = ReplayBuffer(
        args.buffer_size,
        gym.spaces.Box(-1, 1, shape=(968,),dtype=float),
        envs.action_space,
        device,
        handle_timeout_termination=False,
    )

    st = time.time()
    last_step = 0
    """comments: """
    obs, infos = envs.reset()

    episode_length = 0
    episode_reward = 0
    for global_step in range(args.total_timesteps):
        
        """comments: """
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        """comments: """
        if random.random() < epsilon:
            actions = envs.action_space.sample()
        else:
            q_values = q_network(torch.Tensor(obs.flatten()).to(device))
            actions = torch.argmax(q_values, dim=0).cpu().numpy()
        
        """comments: """
        next_obs, rewards, dones, trun, infos = envs.step(actions)

        episode_reward += rewards
        # if global_step > args.learning_starts:
        #     envs.render()
        # envs.render() # close render during training
        # print(actions)
        # print(infos)
        
        # if not infos["rewards"]["on_road_reward"]:
        #     trun = True
        
        if dones or trun:
            logger.info(f"global_step={global_step}, episodic_return={episode_reward}, episodic_length={episode_length}, time_used: {int(time.time() - st)}")
            # print(infos["episode"])
            # logger.info(f"[episode info] env_id: {i}, total_steps: {}, episode lenght: {[i]}, episode reward: [}], time_used: {int(time.time() - st)}")
            writer.add_scalar("charts/episodic_return", episode_reward, global_step)
            writer.add_scalar("charts/episodic_length", episode_length, global_step)
            obs, infos = envs.reset()

            episode_reward = 0
            episode_length = 0
        else:
            episode_length += 1
            obs = next_obs
        """comments: """

        rb.add(obs.flatten(), next_obs.flatten(), actions, rewards, (dones or trun), infos)
        
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            
            """comments: """
            data = rb.sample(args.batch_size)
            
            """comments: """
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            """comments: """
            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
            
            """comments: """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            """comments: """
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

        if global_step % 2000 == 0:
            model_path = os.path.join(out_dir, "models/")
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            torch.save(q_network.state_dict(), model_path + f"dqn.lr_{args.learning_rate}.gamma_{args.gamma}.e_{args.start_e}.ende_{args.end_e}.buffer_{args.buffer_size}.tnf_{args.target_network_frequency}.bs{args.batch_size}.seed_{args.seed}.{time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())}.pth")
            with open (os.path.join(out_dir, "hyperparameters.txt"), 'w') as f: 
                f.write(str(args))

    """close the env and tensorboard logger"""
    envs.close()
    writer.close()