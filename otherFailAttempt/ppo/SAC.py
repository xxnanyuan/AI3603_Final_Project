import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import copy

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std_layer = nn.Linear(hidden_width, action_dim)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)  # We output the log_std to ensure that std=exp(log_std)>0
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)  # Generate a Gaussian distribution
        if deterministic:  # When evaluating，we use the deterministic policy
            a = mean
        else:
            a = dist.rsample()  # reparameterization trick: mean+std*N(0,1)

        if with_logprob:  # The method refers to Open AI Spinning up, which is more stable.
            log_pi = dist.log_prob(a).sum(dim=1, keepdim=True)
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=1, keepdim=True)
        else:
            log_pi = None

        a = self.max_action * torch.tanh(a)  # Use tanh to compress the unbounded Gaussian distribution into a bounded action interval.

        return a, log_pi


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l5 = nn.Linear(hidden_width, hidden_width)
        self.l6 = nn.Linear(hidden_width, 1)

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2
    
    
class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, args, logger, init_e):
        self.max_action = max_action
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.batch_size = args.batchsize  # batch size
        self.GAMMA = args.gamma  # discount factor
        self.TAU = args.tau  # Softly update the target network
        self.q_lr = args.q_lr  # learning rate
        self.policy_lr = args.policy_lr
        self.adaptive_alpha = args.adaptive_alpha  # Whether to automatically learn the temperature alpha
        self.policy_frequency = args.policy_frequency
        self.target_network_frequency = args.target_network_frequency
        self.logger = logger
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -action_dim * init_e
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp().item() * args.alpha
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.q_lr)
        else:
            self.alpha = args.alpha

        '''1. Input: initial policy parameters theta, Q-function parameters phi_1, phi_2'''
        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action)
        self.critic = Critic(state_dim, action_dim, self.hidden_width)

        '''2: Set target parameters equal to main parameters phi_targ1 <- phi_1;  phi_targ2 <- phi_2'''
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.q_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.policy_lr)

    def choose_action(self, s, deterministic=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a, _ = self.actor(s, deterministic, False)  # When choosing actions, we do not need to compute log_pi
        return a.data.numpy().flatten()

    def learn(self, relay_buffer, total_steps):
        '''11. Randomly sample a batch of transitions, B =(s,a, r, s',d) from D'''
        batch_s, batch_a, batch_r, batch_s_, batch_d = relay_buffer.sample(self.batch_size)  # Sample a batch

        '''12. Compute targets for the Q functions:'''
        with torch.no_grad():
            batch_a_, log_pi_ = self.actor(batch_s_)  # a' from the current policy
            # Compute target Q
            target_Q1, target_Q2 = self.critic_target(batch_s_, batch_a_)
            '''在下一步取了min'''
            target_Q = batch_r + self.GAMMA * (1 - batch_d) * (torch.min(target_Q1, target_Q2) - self.alpha * log_pi_)

        '''
        13. Update Q-functions by one step of gradient descent using :
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        '''
        # Compute current Q
        current_Q1, current_Q2 = self.critic(batch_s, batch_a)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute actor loss
        '''
        14. Update policy by one step of gradient ascent using: 
        actor_loss = (self.alpha * log_pi - Q).mean()

        p.s. and update the alpha if it's asked to do so.
        '''
        
        if total_steps % self.policy_frequency == 0:
            for _ in range(self.policy_frequency):
                '''从这里开始更新actor'''
                a, log_pi = self.actor(batch_s)
                Q1, Q2 = self.critic(batch_s, a)
                Q = torch.min(Q1, Q2)
                actor_loss = (self.alpha * log_pi - Q).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                # Update alpha
                if self.adaptive_alpha:
                    # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
                    alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self.alpha = self.log_alpha.exp()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        

        # Softly update target networks
        '''
        15. Update target networks with
        '''
        if total_steps % self.target_network_frequency == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
    
    
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer.pth")
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pth")
        #优化器参数不用存


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pth"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pth"))
        self.actor_target = copy.deepcopy(self.actor)	