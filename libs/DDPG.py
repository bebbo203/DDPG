from agent import Agent
import gym
import numpy as np
from plotter import *
import signal
from matplotlib import pyplot as plt
from gym_cartpole_swingup.envs import CartPoleSwingUpEnv
import os

import sys


actor_checkpoint_path = "training_actor/actor.ckpt"
critic_checkpoint_path = "training_critic/critic.ckpt"
actor_checkpoint_dir = os.path.dirname(actor_checkpoint_path)
critic_checkpoint_dir = os.path.dirname(critic_checkpoint_path)



def save_weights():
    agent.actor_network.save_weights(actor_checkpoint_path.format(epoch=0))
    agent.critic_network.save_weights(critic_checkpoint_path.format(epoch=0))


def signal_handler(sig, frame):
    #plt.close()
    #plt.close()
    #plot_critic_decision(env, agent.actor_network, agent.critic_network)
    #plot_actor_decision(env, agent.actor_network)
   
    #save_weights()

    plt.plot(range(len(mean_rewards)), mean_rewards)
    plt.plot(range(len(rewards)), rewards)
    plt.show()

    sys.exit(0)



signal.signal(signal.SIGINT, signal_handler)
env = gym.make("Pendulum-v0")
#env = CartPoleSwingUpEnv()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

#state_dim = 4
#action_dim = 1

#Agent(self, state_size, action_size, max_action,  minibatch_size, a_lr, c_lr, gamma, tau)
agent = Agent(state_dim, action_dim, action_bound, 128, 0.001, 0.003, 0.99,  0.001)



MAX_EPOCH_TIME = 350
MAX_EPOCHS = 100000
TIME_TO_UPDATE = 100000
TRAINING_INTERVAL = 1
PLOT = False
RENDER = True
EPSILON_MIN = 0.05
EPSILON_START = 1
EPSILON_DECAY = 0.999





total_reward = 0
rewards = []
mean_rewards = []
t_total = 1
epsilon = EPSILON_START

if(PLOT):
    #plot_critic_decision(env, agent.actor_network, agent.critic_network)
    heatmap(env, agent.actor_network)


for i in range(MAX_EPOCHS):
    s = env.reset()
    
   
    
    total_reward = 0
    epoch_duration = 0
    for t_ in range(MAX_EPOCH_TIME):
        if(RENDER and t_total > TIME_TO_UPDATE):
          env.render()
        
        
        if(t_total < TIME_TO_UPDATE):
            a = env.action_space.sample()
        else:
            pure_action, noisy_action = agent.act(s, t_total)
            
            if(epsilon > EPSILON_MIN and np.random.normal() < epsilon):
                a = noisy_action
            else:
                a = pure_action
                #print(str(pure_action)+"  "+str(noisy_action))
        
            epsilon = epsilon if epsilon <= EPSILON_MIN else epsilon * EPSILON_DECAY



        s_1, r, t, _ = env.step(a)
        if(t_ == MAX_EPOCH_TIME - 1):
            t = 1
            

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
        
        

    print("%3d  t_r: %8.3f     m_r: %6.4f    frames: %4d  epsilon: %f" % (i, total_reward, total_reward/epoch_duration,epoch_duration, epsilon))

    
    
    if(PLOT and i % 2 == 0 and t_total > TIME_TO_UPDATE):
        plt.close()
        #plt.close()
        #plot_critic_decision(env, agent.actor_network, agent.critic_network)
        heatmap(env, agent.actor_network)
    
    
    
    


           
    

