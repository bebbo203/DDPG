from agent import Agent
import gym
import numpy as np
from plotter import *
import signal
from matplotlib import pyplot as plt

import sys





def signal_handler(sig, frame):
    #plt.close()
    #plt.close()
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
agent = Agent(state_dim, action_dim, 64, 0.0001, 0.001, 0.99,  0.001)



MAX_EPOCH_TIME = 5000
MAX_EPOCHS = 500000
TIME_TO_UPDATE = 100000
TRAINING_INTERVAL = 10
PLOT = True
EPSILON_MIN = 0.1
EPSILON_START = 1
EPSILON_DECAY = 0.99



total_reward = 0
rewards = []
mean_rewards = []
t_total = 1
epsilon = EPSILON_START

if(PLOT):
    plot_critic_decision(env, agent.actor_network, agent.critic_network)
    plot_actor_decision(env, agent.actor_network)


for i in range(MAX_EPOCHS):
    s = env.reset()
    
   
    
    total_reward = 0
    epoch_duration = 0
    for t_ in range(MAX_EPOCH_TIME):
        #env.render()
        
        
        if(t_total < TIME_TO_UPDATE):
            a = env.action_space.sample()
        else:
            pure_action, noisy_action = agent.act(s, t_total)
            a = noisy_action


        # if(t_total < TIME_TO_UPDATE):
        #     a = noisy_action
        # else:
        #     a = pure_action
        # if(np.random.rand() < epsilon):
        #     a = noisy_action
        # else:
        #     a = pure_action

        # if(epsilon >= EPSILON_MIN):
        #     epsilon *= EPSILON_DECAY
        
        
        s_1, r, t, _ = env.step(a)
        total_reward += r
        
        s_1 = np.array(s_1).reshape(state_dim)
        
        agent.step(s, a, r, s_1, t,  t_total>=TIME_TO_UPDATE and t_total % TRAINING_INTERVAL == 0)
        if(t_total==TIME_TO_UPDATE):
            print("TRAINING PHASE")
       
        epoch_duration +=1
        s = s_1
        t_total += 1
        if(t):
            if(t_total >= TIME_TO_UPDATE):
                mean_rewards.append(total_reward/epoch_duration)
                rewards.append(total_reward)
            break
        
        

    print("%3d:  %3f    %3f      %d" % (i, total_reward, total_reward/epoch_duration,epoch_duration))

    
    
    if(PLOT and i % 2 == 0 and t_total > TIME_TO_UPDATE):
        plt.close()
        plt.close()
        plot_critic_decision(env, agent.actor_network, agent.critic_network)
        plot_actor_decision(env, agent.actor_network)
    
    
    
    


           
    

