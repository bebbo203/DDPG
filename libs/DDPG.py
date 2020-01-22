from agent import Agent
import gym
import numpy as np
from plotter import *
import signal
from matplotlib import pyplot as plt

import sys





def signal_handler(sig, frame):
    plt.close()
    plt.close()
    #plot_critic_decision(env, agent.actor_network, agent.critic_network)
    #plot_actor_decision(env, agent.actor_network)
    
    plt.plot(range(len(mean_rewards)), mean_rewards)
    plt.plot(range(len(rewards)), rewards)
    plt.draw()
    sys.exit(0)



signal.signal(signal.SIGINT, signal_handler)
env = gym.make("MountainCarContinuous-v0")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high

#Agent(self, state_size, action_size, minibatch_size, a_lr, c_lr, gamma, tau)
agent = Agent(state_dim, action_dim, 64, 0.0001, 0.0005, 0.99,  0.001)



MAX_EPOCH_TIME = 500
MAX_EPOCHS = 500000
PLAY_AFTER = 1



total_reward = 0
rewards = []
mean_rewards = []
t_total = 1

plot_critic_decision(env, agent.actor_network, agent.critic_network)
plot_actor_decision(env, agent.actor_network)


for i in range(MAX_EPOCHS):
    s = env.reset()
    
   
    
    total_reward = 0
    epoch_duration = 0
    for t_ in range(MAX_EPOCH_TIME):
        #env.render()

        
        pure_action, noisy_action = agent.act(s, t_total)
        a = noisy_action
        

        s_1, r, t, _ = env.step(a)
        total_reward += r
        
        s_1 = np.array(s_1).reshape(state_dim)
        
        

        agent.step(s, a, r, s_1, t)
       
        epoch_duration +=1
        s = s_1
        if(t):
            mean_rewards.append(total_reward/epoch_duration)
            rewards.append(total_reward)
            break
        
        t_total += 1

    print("%3d:  %3f    %f" % (i, total_reward, total_reward/epoch_duration))

    plt.close()
    plt.close()
    
    plot_critic_decision(env, agent.actor_network, agent.critic_network)
    plot_actor_decision(env, agent.actor_network)
    
    
    
    


sys.exit(0)    
    

