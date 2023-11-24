# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy

import os
import random
import time
from distutils.util import strtobool
import gymnasium
import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from utils import get_logger


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Hopper-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500,
        help="total timesteps of the experiments")

    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=48,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=0,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args


def make_highway_env(seed):
    def thunk():
        env = gymnasium.make("highway-v0", render_mode="rgb_array")
        env.configure({
            "observation": {
            "type": "OccupancyGrid",
            "vehicles_count": 15,
            "features": ["presence", "on_road","x", "y", "vx", "vy", "cos_h", "sin_h"],
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
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
            },

            "duration": 40,  # [s]
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 1,  # [Hz]
        })
        env.reset()
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        #env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


'''
args: Namespace(alpha=0.2, autotune=True, batch_size=256, buffer_size=1000000, capture_video=False, cuda=True, 
env_id='Hopper-v4', exp_name='sac_continuous_action', gamma=0.99, learning_starts=5000.0, noise_clip=0.5, 
policy_frequency=2, policy_lr=0.0003, q_lr=0.001, seed=1, target_network_frequency=1, tau=0.005, torch_deterministic=True, 
total_timesteps=1000000, track=False, wandb_entity=None, wandb_project_name='cleanRL')
'''


number = 1
times = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
out_dir = f"train/{times}"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
log_dir = os.path.join(out_dir, 'train.log')
logger = get_logger(log_dir, name="log", level="info")



import stable_baselines3 as sb3

if sb3.__version__ < "2.0":
    raise ValueError(
        """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
    )

args = parse_args()
# logger.info(args)
run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
'''#track = false,所以这个if没用'''
if args.track:    
    import wandb

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )

writer = SummaryWriter(os.path.join(out_dir, f"{run_name}"))
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

# TRY NOT TO MODIFY: seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

# env setup
envs = gym.vector.SyncVectorEnv(
    [make_highway_env(args.seed)]
)
# logger.info(envs)
'''这行没用'''
assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"


'''最大动作空间维度'''

# max_action = float(envs.single_action_space.high[0])

# print(envs.single_observation_space.shape)

actor = Actor(envs).to(device)
qf1 = SoftQNetwork(envs).to(device)
qf2 = SoftQNetwork(envs).to(device)
qf1_target = SoftQNetwork(envs).to(device)
qf2_target = SoftQNetwork(envs).to(device)
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())
q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

# Automatic entropy tuning
'''autotune=True，表示我们会自动调节最大熵的权重函数'''
if args.autotune:
    target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    '''表示我们要通过这个optimizer来更新alpha.'''
    a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
else:
    alpha = args.alpha

# logger.info("obs_space: ",envs.single_observation_space)
# logger.info("action_space: ",envs.single_action_space)
# '''
# obs_space:  Box(-inf, inf, (8, 11, 11), float32)
# action_space:  Box(-1.0, 1.0, (2,), float32)
# '''
envs.single_observation_space.dtype = np.float32
'''buffer_size=1000000, 好大...'''
rb = ReplayBuffer(
    args.buffer_size,
    envs.single_observation_space,
    envs.single_action_space,
    device,
    handle_timeout_termination=False,
)
start_time = time.time()

# TRY NOT TO MODIFY: start the game
obs, _ = envs.reset(seed=args.seed)


# flatten the batch
'''
b_logprobs = logprobs.reshape(-1)
b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
b_advantages = advantages.reshape(-1)
b_returns = returns.reshape(-1)
b_values = values.reshape(-1)

'''
#flatten the batch
b_obs = obs.reshape(1, np.prod(obs.shape))
print(type(b_obs))

st = time.time()

for global_step in range(args.total_timesteps):
    # ALGO LOGIC: put action logic here
    # if global_step % 100 == 0:
    '''learning_starts=5000.0'''
    if global_step < args.learning_starts:
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    else:
        actions, _, _ = actor.get_action(torch.Tensor(b_obs).to(device))
        actions = actions.detach().cpu().numpy()

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, rewards, terminations, truncations, infos = envs.step(actions)

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    if "final_info" in infos:
        for info in infos["final_info"]:
            log_time = time.time()
            logger.info(f"global_step={global_step}, episodic_return={info['episode']['r']}, time: {int(log_time - st)}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            break

    # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
    real_next_obs = next_obs.copy()
    for idx, trunc in enumerate(truncations):
        if trunc:
            real_next_obs[idx] = infos["final_observation"][idx]
    rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    obs = next_obs

    # ALGO LOGIC: training.
    '''实际上这里和上面的 glbl_stp < args.lrning_starts 分为两支，前面一支应该是随机探索收集buffer_size，后面一支应该是在学'''
    if global_step > args.learning_starts:
        data = rb.sample(args.batch_size)
        '''flatten the data'''
        # print(data.next_observations.shape)
        b_observations = data.next_observations.reshape(args.batch_size, int(np.prod(data.next_observations.shape[1:])))
        with torch.no_grad():
            
            next_state_actions, next_state_log_pi, _ = actor.get_action(b_observations)
            qf1_next_target = qf1_target(b_observations, next_state_actions)
            qf2_next_target = qf2_target(b_observations, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

        
        # if global_step % 100 == 0:
        #     logger.info(f"reward: {np.mean(data.rewards.flatten())}")
        

        qf1_a_values = qf1(b_observations, data.actions).view(-1)
        qf2_a_values = qf2(b_observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        '''实际更新网络参数的代码在这里，注意这里只更新了q_optimizer!论文里面讲到actor和Q的更新是不同步的'''
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        '''满足这个条件才会更新policy,也就是actor'''
        '''policy_frequency=2'''
        if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
            '''em应该是说每freq次才更新，每次更新都更新policy次，所以是延迟更新，但是更新的次数还是相等的'''
            for _ in range(
                args.policy_frequency
            ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                pi, log_pi, _ = actor.get_action(b_observations)
                qf1_pi = qf1(b_observations, pi)
                qf2_pi = qf2(b_observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                '''从这里开始更新actor'''
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                '''autotune=True，这个if下面都是自动调整alpha参数的部分'''
                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(b_observations)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

        # update the target networks
        '''
        如果满足target_freq的话就更新target_work
        target_freq = 1     
        这是软更新，即让目标网络每一步都更新。这里是将param_data和tau那个那个公式相乘以后赋值给target.

        '''
        if global_step % args.target_network_frequency == 0:
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        '''
        它这个不出reward值的，所以不太好说save_best, 我先存最后一个，如果work我就存中间的
        以下是存网络的过程，我只存了pth，到时候要load_dict()    
        '''
        if global_step == args.total_timesteps - 1: 
            logger.info("saving, plz wait...")
            if args.autotune:
                alpha_dir = os.path.join(out_dir, "alpha.txt")
                with open(alpha_dir, 'w') as f:
                    f.write(str(alpha))
            torch.save(qf1.state_dict(), out_dir + "/qf1.pth")
            torch.save(qf2.state_dict(), out_dir + "/qf2.pth")
            torch.save(actor.state_dict(), out_dir + "/actor.pth")


        '''这个下面都是log的东西，不用看'''
        if global_step % 100 == 0:
            writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/alpha", alpha, global_step)
            logger.info("SPS:" + str(int(global_step / (time.time() - start_time))))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            if args.autotune:
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

envs.close()
writer.close()
