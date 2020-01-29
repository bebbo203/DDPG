from agent import Agent
import gym
import numpy as np
from plotter import *
import signal
from matplotlib import pyplot as plt
import os
import sys


actor_checkpoint_path = "training_actor/actor.ckpt"
critic_checkpoint_path = "training_critic/critic.ckpt"
actor_checkpoint_dir = os.path.dirname(actor_checkpoint_path)
critic_checkpoint_dir = os.path.dirname(critic_checkpoint_path)


def save_weights():
    agent.actor_network.save_weights(actor_checkpoint_path)
    agent.critic_network.save_weights(critic_checkpoint_path)
    print("Models saved!")


def signal_handler(sig, frame):   
    save_weights()

    plt.plot(range(len(mean_rewards)), mean_rewards)
    plt.plot(range(len(rewards)), rewards)
    plt.show()

    sys.exit(0)



signal.signal(signal.SIGINT, signal_handler)
env = gym.make("LunarLanderContinuous-v2")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

agent = Agent(state_dim, action_dim, action_bound, 64, 0.001, 0.0001, 0.99,  0.001)


MAX_EPOCH_TIME = 1000
MAX_EPOCHS = 100000
TIME_TO_UPDATE = 100000
TRAINING_INTERVAL = 1
RENDER = False
EPSILON_MIN = 0.07
EPSILON_START = 1.5
EPSILON_DECAY = 0.9998





total_reward = 0
rewards = []
mean_rewards = []
t_total = 1
epsilon = EPSILON_START


for i in range(MAX_EPOCHS):
    s = env.reset()
    
   
    total_reward = 0
    epoch_duration = 0
    for t_ in range(MAX_EPOCH_TIME):
        if(RENDER and t_total > TIME_TO_UPDATE):
            env.render()
          
        if(t_total < TIME_TO_UPDATE):
            _, a = agent.act(s, t_total)
        else:
            pure_action, noisy_action = agent.act(s, t_total)
            
            if(epsilon >= EPSILON_MIN and np.random.normal() < epsilon):
                a = noisy_action
            else:
                a = pure_action
        
            epsilon = epsilon if epsilon < EPSILON_MIN else epsilon * EPSILON_DECAY

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


    
    
    


           
    

