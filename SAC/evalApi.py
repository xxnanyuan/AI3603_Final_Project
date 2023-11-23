from SAC import SAC
import numpy as np
agent = False
def getAction(env_name, env, obs):
    global agent
    if agent == False:
        state_dim = np.prod(env.observation_space.shape)
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        agent = SAC(state_dim, action_dim, max_action)
        agent.load(f"./models/{env_name}")
    obs = obs.flatten()
    action = agent.choose_action(obs)
    return action
    
    