from actor import Actor
from critic import Critic
from replayBuffer import ReplayBuffer
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import gym
import numpy as np
import signal
import sys

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

def signal_handler(sig, frame):
    plt.plot(range(len(rewards)), rewards)
    plt.show()
    sys.exit(0)


EPISODES = 500
MINIBATCH_SIZE = 32
BUFFER_SIZE = 10000
TRAIN_AFTER = MINIBATCH_SIZE + 1
T_MAX = 200
G = 0.98
UPDATE_FREQ = 4
EPSILON_START = 1.0
EPSILON_STOP = 0.1
EPSILON_DECAY = 0.99


signal.signal(signal.SIGINT, signal_handler)
env = gym.make("MountainCarContinuous-v0")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high

actor = Actor(state_dim, action_dim, action_bound, MINIBATCH_SIZE, 0.0001, 0.05)
critic = Critic(state_dim, action_dim, action_bound, MINIBATCH_SIZE, 0.0001, 0.05)
replay = ReplayBuffer(BUFFER_SIZE)
actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))


rewards = []

epsilon = EPSILON_START

for n_episode in range(EPISODES):
    s = env.reset()
    s = np.array([s.flatten().astype(dtype="float32")])
    

    d = False
    t = 0
    total_r = 0
    final_t = 0
    for t in range(T_MAX):
        env.render()
        

        #Epsilon-greedy
        if(np.random.rand() < epsilon):
            a = [env.action_space.sample()]
        else:
            a = actor.predict(s) + actor_noise()
        if(epsilon >= EPSILON_STOP):
            epsilon *= EPSILON_DECAY
        
        
        
        s_next, r, d, _ = env.step(a)
        s_next = np.array([s_next.flatten().astype(dtype="float32")])
        total_r += r
        replay.add(s, a, r, s_next, d)
        s = s_next

        
        if(replay.size() >= TRAIN_AFTER):

            s_batch, a_batch, r_batch, s_1_batch, d_batch = \
                replay.sample_batch(MINIBATCH_SIZE)

            target_q = critic.predict_target(s_1_batch, actor.predict_target(s_1_batch), axis=2)
           
            y = np.array([])
            for i in range(MINIBATCH_SIZE):
                if(d_batch[i] == False):
                    y = np.append(y,r_batch[i] + G*target_q[i])
                else:
                    y = np.append(y,r_batch[i])
                
               
            
            #Update critic
            critic.train(s_batch, a_batch, y, axis=2)

            #Update actor
            a_outs = actor.predict(s_batch) 
            critic_grads = critic.get_critic_gradient(s_batch, a_outs)
            actor.train(s_batch, critic_grads)

        
            actor.update_target_network()
            critic.update_target_network()

        final_t = t
        if(d == True):
            break
    
    
    rewards.append(total_r)
    print("%d:\t%d\t%f" % (n_episode,final_t, total_r))
        

