import tensorflow as tf
from functools import reduce

class Actor(object):

    def __init__(self,state_size, action_size, action_bounds, batch_size, lr, tau):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.action_bounds = action_bounds
        self.lr = lr
        self.tau = tau

        self.opt = tf.optimizers.SGD(self.lr)
        self.initializer = tf.initializers.glorot_uniform()
        
        self.shapes = [
            [self.state_size, 200],
            [200],
            [200, self.action_size],
            [self.action_size]
        ]

        self.weights = []
        self.target_weights = []
        
        for i in range(len(self.shapes)):
            self.weights.append(self.init_weights(self.shapes[i],'AW{}'.format(i)))
            self.target_weights.append(self.init_weights(self.shapes[i],'ATW{}'.format(i)))

        #Should be given by Critic
        #self.critic_gradient = tf.placeholder(tf.float32, [None, self.action_size])
        #self.total_gradient = tf.gradient(self.out, self.params, -self.critic_gradient)
        #self.gradient = list(map(lambda x: tf.div(x, self.batch_size), total_gradient))

        #self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.gradient, self.params))

    def dense(self, inputs, weights, biases):
        x = tf.add(tf.matmul(inputs, weights), biases)
        return x

    def init_weights(self, shape, name):
        return tf.Variable(self.initializer(shape), name=name, trainable=True, dtype=tf.float32)

    #mu: S -> A
    def network(self, states):
        d1 = self.dense(states, self.weights[0], self.weights[1])
        d1 = tf.nn.relu(d1)
        d2 = self.dense(d1, self.weights[2], self.weights[3])
        d2 = tf.nn.tanh(d2)
        return d2

    def target_network(self, states):
        dt1 = self.dense(states, self.target_weights[0], self.target_weights[1])
        dt1 = tf.nn.relu(dt1)
        dt2 = self.dense(dt1, self.target_weights[2], self.target_weights[3])
        dt2 = tf.nn.tanh(dt2)
        return dt2

    
    
    def train(self, states, critic_gradient):
        with tf.GradientTape() as t:
            actor_pred = self.network(states)

        actor_gradient = \
            t.gradient(actor_pred, self.weights, -critic_gradient)
        
        self.opt.apply_gradients(zip(actor_gradient, self.weights))
        #opt.minimize(actor_gradient, self.weights)

    #OH MOLTO PROBABILMENTE DEVI DIVIDERE EH
    def train_batch(self, states, critic_gradients):
       
        with tf.GradientTape() as t:
            actor_pred = self.network(states)

        #METTERE IL MENO UNO ALLA FINE
        actor_gradients = t.gradient(actor_pred, self.weights, -1*critic_gradients)
            
        
        self.opt.apply_gradients(zip(actor_gradients, self.weights))
        #opt.minimize(actor_gradients, var_list = self.weights)

            
        #optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.gradient, self.weights))

    # def predict(self, states):
    #     self.sess.run(self.out, feed_dict = {self.x: states})

    def predict(self, states):
        return self.network(states)

    def predict_target(self, states):
        return self.target_network(states)

    def update_target_network(self):
        for i in range(len(self.target_weights)):
            self.target_weights[i] = tf.multiply(self.weights[i], self.tau) \
                + tf.multiply(self.target_weights[i], 1. - self.tau)

    
        

    # def predit_target(self, states):
    #     self.sess.run(self.target_out, feed_dict = {self.target_x: states})

    # def update_target_network(self):
    #     for i in range(len(self.target_params)):
    #         self.target_params[i].assign(tf.multiply(self.params[i], self.tau) \
    #             + tf.multiply(self.target_params[i], 1. - self.tau))

    # def num_vars(self):
    #     return len(self.params) + len(self.target_params)
    
