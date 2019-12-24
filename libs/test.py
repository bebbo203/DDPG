from actor import Actor
from critic import Critic
from replayBuffer import ReplayBuffer
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import gym
import numpy as np

state_dim = 3
action_dim = 1
EPISODES = 500
MINIBATCH_SIZE = 64
BUFFER_SIZE = 500
T_MAX = 2100
G = 0.01

#C'è un problema nel numero di t_max che se finisce per colpa di d allora non va più

env = gym.make("Pendulum-v0")
actor = Actor(state_dim, action_dim, 0, MINIBATCH_SIZE, 0.001, 0.1)
critic = Critic(state_dim, action_dim, 0, MINIBATCH_SIZE, 0.001, 0.1)
replay = ReplayBuffer(BUFFER_SIZE)


for n_episode in range(EPISODES):
    s = env.reset()
    s = np.array([s.flatten().astype(dtype="float32")])
    

    d = False
    t = 0
    total_r = 0
    for t in range(T_MAX):
        env.render()
        a = actor.predict(s)
        
        
        s_next, r, d, _ = env.step(a)
        s_next = np.array([s_next.flatten().astype(dtype="float32")])
        total_r += r
        replay.add(s, a, r, s_next, d)
        s = s_next

        

        if(d == True):
            break
        if(replay.size() >= MINIBATCH_SIZE):

            s_batch, a_batch, r_batch, s_1_batch, d_batch = \
                replay.sample_batch(MINIBATCH_SIZE)

            #Critic Train
            y = np.array([])
            for i in range(MINIBATCH_SIZE):
                predicted_action = actor.predict_target(s_1_batch[i])
                predicted_Q = critic.predict_target(s_1_batch[1], predicted_action)
               

                print(predicted_Q)

                if(d_batch[i] == False):
                    y = np.append(y,r_batch[i] + G*predicted_Q)
                else:
                    y = np.append(y,r_batch[i])
                    
            

            loss = critic.train(s_batch, a_batch, y, axis=2)
             
            critic_grads = critic.get_critic_gradient(s_batch, a_batch)
            
            actor.train_batch(s_batch, critic_grads)

            actor.update_target_network()
            critic.update_target_network()

        
     
            
        
         
        
        
        
