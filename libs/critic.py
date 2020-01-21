import tensorflow as tf
import numpy as np


W1 = 50
W2 = 50

class Critic(object):

    #MUST be initialized AFTER actor
    def __init__(self, state_size, action_size, action_bounds, batch_size, lr, tau):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.lr = lr
        self.tau = tau

        self.opt = tf.optimizers.Adam(self.lr)
        self.initializer = tf.initializers.GlorotUniform()



        #DEVONO PARTIRE CON GLI STESSI PESIIIIIIII
        self.shapes = [
            [self.state_size+self.action_size, W1],
            [W1],
            [W1, W2],
            [W2],
            [W2, 1],
            [1]
        ]
        
        self.weights = []
        self.target_weights = []

        for i in range(len(self.shapes)):
            if(i == len(self.shapes)-1 or i == len(self.shapes)-2):
                new_weight = tf.Variable(tf.random.uniform(self.shapes[i],-3e-3,3e-3))
            else:
               new_weight = self.init_weights(self.shapes[i], 'CW{}'.format(i))
            self.weights.append(new_weight)
            self.target_weights.append(new_weight)
            
        


    def dense(self, inputs, weights, biases):
        x = tf.add(tf.matmul(inputs, weights), biases)
        return x

    def init_weights(self, shape, name, f=0):
        return tf.Variable(self.initializer(shape), name=name, trainable=True, dtype=tf.float32)
        
    #Q: A,S -> q
    def _network(self, states, actions, weights, axis):
        x = tf.concat([states, actions], axis=axis)
        x = x/tf.reduce_max(tf.abs(x))
        d1 = self.dense(x, weights[0], weights[1])
        d1 = tf.nn.relu(d1)
        d2 = self.dense(d1, weights[2], weights[3])
        d2 = tf.nn.relu(d2)
        d3 = self.dense(d2, weights[4], weights[5])
        d3 = tf.identity(d3)
        return d3

    def target_network(self, states, actions, axis=1):
        return self._network(states, actions, self.target_weights, axis)

    def network(self, states, actions, axis=1):
        return self._network(states, actions, self.weights, axis)
        
    def train(self, states, actions, y, axis=1): 
        cost = lambda: tf.reduce_mean(tf.square(y - self.predict(states, actions, axis=2)))
        self.opt.minimize(cost, self.weights) 

    def predict(self, states, actions, axis=1):
        return self.network(states, actions, axis)

    def predict_target(self, states, actions, axis=1):
        return self.target_network(states,actions, axis)

    def get_critic_gradient(self, states, actions):
        actions = tf.constant(actions, dtype="float32")
        with tf.GradientTape() as t:
            t.watch(actions)
            value = self.network(states, actions, axis=2)
        grad = t.gradient(value, actions)
        return grad

    def update_target_network(self):
        for i in range(len(self.target_weights)):
            self.target_weights[i] = tf.multiply(self.weights[i], self.tau) \
                + tf.multiply(self.target_weights[i], 1. - self.tau)
