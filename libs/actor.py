import tensorflow as tf
from functools import reduce


W1 = 50
W2 = 50


class Actor(object):

    def __init__(self,state_size, action_size, action_bounds, batch_size, lr, tau):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.action_bounds = action_bounds
        self.lr = lr
        self.tau = tau

        self.opt = tf.optimizers.Adam(self.lr)
        self.initializer = tf.initializers.GlorotUniform()
        
        self.shapes = [
            [self.state_size, W1],
            [W1],
            [W1, W2],
            [W2],
            [W2, self.action_size],
            [self.action_size]
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

    def init_weights(self, shape, name):
        return tf.Variable(self.initializer(shape), name=name, trainable=True, dtype=tf.float32)

    #mu: S -> A
    def _network(self, states, weights):
        states = states/tf.reduce_max(tf.abs(states))
        d1 = self.dense(states, weights[0], weights[1])
        d1 = tf.nn.relu(d1)
        d2 = self.dense(d1, weights[2], weights[3])
        d2 = tf.nn.relu(d2)
        d3 = self.dense(d2, weights[4], weights[5])
        d3 = tf.nn.tanh(d3)
        return d3*self.action_bounds

    def network(self, states):
        return self._network(states, weights=self.weights)

    def target_network(self, states):
        return self._network(states, weights=self.target_weights)
        

    def train(self, states, critic_gradients):
        with tf.GradientTape() as t:
            actor_pred = self.network(states)

        actor_gradients = \
            t.gradient(actor_pred, self.weights, -1*critic_gradients)
        actor_gradients = list(map(lambda x: x/self.batch_size, actor_gradients))
        
        self.opt.apply_gradients(zip(actor_gradients, self.weights))
    

    def predict(self, states):
        return self.network(states)

    def predict_target(self, states):
        return self.target_network(states)

    def update_target_network(self):
        for i in range(len(self.target_weights)):
            self.target_weights[i] = tf.multiply(self.weights[i], self.tau) \
                + tf.multiply(self.target_weights[i], 1. - self.tau)
