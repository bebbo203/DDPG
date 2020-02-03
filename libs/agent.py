from critic import Critic
from actor import Actor
from replayBuffer import ReplayBuffer
import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np
from ounoise import OUNoise
import functools




class Agent(object):
    def __init__(self, state_size, action_size, max_action, minibatch_size, a_lr, c_lr, gamma, tau):
        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action
        
        self.critic_lr = c_lr
        self.actor_lr = a_lr

        self.actor_network = Actor(self.state_size, self.action_size, self.max_action, self.actor_lr)
        self.actor_target_network = Actor(self.state_size, self.action_size, self.max_action, self.actor_lr)
        self.critic_network = Critic(self.state_size, self.action_size, self.critic_lr)
        self.critic_target_network = Critic(self.state_size, self.action_size, self.critic_lr)

        self.actor_target_network.set_weights(self.actor_network.get_weights())
        self.critic_target_network.set_weights(self.critic_network.get_weights())

        self.critic_optimizer = optimizers.Adam(learning_rate=self.critic_lr)
        self.actor_optimizer = optimizers.Adam(learning_rate=self.actor_lr)

        self.replay_buffer = ReplayBuffer(1e6)
        self.MINIBATCH_SIZE = minibatch_size
        self.GAMMA = tf.cast(gamma, dtype=tf.float64)
        self.TAU = tau
        self.noise = OUNoise(self.action_size)

        

    
    def step(self, s, a, r, s_1, t, train=True)  
        self.replay_buffer.add(s,a,r,s_1,t)
        if(train and self.replay_buffer.size() >= self.MINIBATCH_SIZE):
            minibatch = self.replay_buffer.sample_batch(self.MINIBATCH_SIZE)
            self.learn(minibatch)

    @tf.function
    def critic_train(self, minibatch):
        s_batch, a_batch, r_batch, s_1_batch, t_batch = minibatch

        mu_prime = self.actor_target_network(s_1_batch)
        q_prime = self.critic_target_network([s_1_batch, mu_prime])

        ys = r_batch + self.GAMMA * (1 - t_batch) * q_prime

        with tf.GradientTape() as tape:    
            predicted_qs = self.critic_network([s_batch, a_batch])
            loss = (predicted_qs - ys)*(predicted_qs - ys) 
            loss = tf.reduce_mean(loss)
        dloss = tape.gradient(loss, self.critic_network.trainable_weights)

        self.critic_optimizer.apply_gradients(zip(dloss, self.critic_network.trainable_weights))
        

    
    def actor_train(self, minibatch):
        s_batch, _, _, _, _ = minibatch

        with tf.GradientTape() as tape:
            next_action = self.actor_network(s_batch)
            actor_loss = -tf.reduce_mean(self.critic_network([s_batch, next_action]))
        actor_grad = tape.gradient(actor_loss, self.actor_network.trainable_weights)
        
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_network.trainable_weights))


    def learn(self, minibatch):
        s, a, r, s_1, t = minibatch

        s = np.array(s, dtype=np.float64).reshape(self.MINIBATCH_SIZE, self.state_size)
        s = tf.convert_to_tensor(s)
        a = np.array(a, dtype=np.float64).reshape(self.MINIBATCH_SIZE, self.action_size)
        a = tf.convert_to_tensor(a)
        r = np.array(r, dtype=np.float64).reshape(self.MINIBATCH_SIZE, 1)
        s_1 = np.array(s_1, dtype=np.float64).reshape(self.MINIBATCH_SIZE, self.state_size)
        s_1 = tf.convert_to_tensor(s_1)
        t = np.array(t, dtype=np.float64).reshape(self.MINIBATCH_SIZE, 1)
        
        minibatch = (s, a, r, s_1, t)
    
        self.critic_train(minibatch)
        self.actor_train(minibatch)       
        self.update_target_networks()

    def act(self, state, t=0):
        state = np.array(state).reshape(1, self.state_size)
        action = self.actor_network(state)[0]
        noisy = self.noise.get_action(action, t)
        return action, noisy

   
    def update_target_networks(self):
        self.actor_target_network.set_weights(np.array(self.actor_network.get_weights())*self.TAU + np.array(self.actor_target_network.get_weights())*(1-self.TAU))
        self.critic_target_network.set_weights(np.array(self.critic_network.get_weights())*self.TAU + np.array(self.critic_target_network.get_weights())*(1-self.TAU))


