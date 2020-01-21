from agent import Agent
import gym
import numpy as np
from plotter import *
import signal
from matplotlib import pyplot as plt
import sys





def signal_handler(sig, frame):
    plot_critic_decision(env, agent.actor_network, agent.critic_network)
    plot_actor_decision(env, agent.actor_network, agent.critic_network)
    rewards.remove(0)
    plt.plot(range(len(rewards)), rewards)
    plt.show()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
env = gym.make("MountainCarContinuous-v0")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high

#Agent(self, state_size, action_size, minibatch_size, lr, gamma, tau)
agent = Agent(state_dim, action_dim, 64, 0.005, 0.99, 0.001)



MAX_EPOCH_TIME = 500
MAX_EPOCHS = 50000
EPSILON_START = 1
EPSILON_END = 0.01
EPSILON_DECAY = 0.9998



total_reward = 0
rewards = []


epsilon = EPSILON_START
for i in range(MAX_EPOCHS):
    s = env.reset()
    rewards.append(total_reward)
    total_reward = 0

    for t in range(MAX_EPOCH_TIME):
        env.render()

        pure_action, noisy_action = agent.act(s) * action_bound

        if(np.random.rand() < epsilon):
            a = noisy_action
        else: 
            a = pure_action
        if(epsilon > EPSILON_END):
            epsilon *= EPSILON_DECAY
        
        
        s_1, r, t, _ = env.step(a)
        total_reward += r
        s_1 = np.array(s_1).reshape(state_dim)
        agent.step(s, a, r, s_1, t)
        
        if(t):
            break


        s = s_1

    print("%3d:  %3f    %f" % (i, total_reward, epsilon))
plot_critic_decision(env, agent.actor_network, agent.critic_network)

plt.plot(range(len(rewards)), rewards)
plt.show()
sys.exit(0)    
    

