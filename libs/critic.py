import tensorflow as tf
import numpy as np

class Critic(object):

    #MUST be initialized AFTER actor
    def __init__(self, state_size, action_size, action_bounds, batch_size, lr, tau):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.lr = lr
        self.tau = tau

        self.opt = tf.optimizers.SGD(self.lr)
        self.initializer = tf.initializers.glorot_uniform()

        self.shapes = [
            [self.state_size+self.action_size, 200],
            [200],
            [200, 1],
            [1]
        ]
        
        self.weights = []
        self.target_weights = []

        for i in range(len(self.shapes)):
            self.weights.append(self.init_weights(self.shapes[i],'CW{}'.format(i)))
            self.target_weights.append(self.init_weights(self.shapes[i],'CTW{}'.format(i)))
        
        # self.s, self.a, self.out = create_network()
        # self.params = tf.trainable_variables()[len(self.num_actor_vars):]
        # self.target_s, self.target_a,  self.target_out = create_network()
        # self.target_params = \
        #     tf.trainable_variables()[len(self.num_actor_vars)+len(self.params):]

        # #That should be the Bellman's equation results
        # self.y = tf.placeholder(tf.float32, [None, 1])

        # self.loss = tf.losses.mean_squared_error(self.y, self.out)

        # self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # self.critic_gradient = tf.gradients(self.out, self.action)

    def dense(self, inputs, weights, biases):
        x = tf.add(tf.matmul(inputs, weights), biases)
        return x

    def init_weights(self, shape, name):
        return tf.Variable(self.initializer(shape), name=name, trainable=True, dtype=tf.float32)
        
    #Q: A,S -> q
    def network(self, states, actions, axis):
        x = tf.concat([states, actions], axis=axis)
        d1 = self.dense(x, self.weights[0], self.weights[1])
        d1 = tf.nn.relu(d1)
        d2 = self.dense(d1, self.weights[2], self.weights[3])
        d2 = tf.nn.relu(d2)
        return d2

    def target_network(self, states, actions):
        x = tf.concat([states, actions], 1)
        d1 = self.dense(x, self.target_weights[0], self.weights[1])
        d1 = tf.nn.relu(d1)
        d2 = self.dense(d1, self.target_weights[2], self.weights[3])
        d2 = tf.nn.relu(d2)
        return d2

    
    def loss(self, truth, predicted):
        return tf.losses.mean_squared_error(truth, predicted)

        
    def train(self, states, actions, y, axis=1):
        loss = lambda:tf.losses.mean_squared_error(y, self.predict(states, actions, axis=axis))
        self.opt.minimize(loss, self.weights)
        return tf.losses.mean_squared_error(y, self.predict(states, actions, axis=axis))

    # def train_batch(self, states, actions, y):
    #     total_loss = 0
    #     for i in range(self.batch_size):
    #         with tf.GradientTape() as t:
    #             current_loss = self.loss(y, self.predict(states[i], actions[i]))
    #             total_loss += current_loss
    #             total_loss = total_loss / self.batch_size
    #     grads = t.gradient(total_loss, self.weights)
    #     opt = tf.optimizers.Adam(self.lr)
    #     #opt.apply_gradients(zip(grads, self.weights))
    #     opt.minimize(self.loss(), self.weights)
    #     return total_loss

    #CORREGGIIIIII
    #deprecated
    def train_batch(self, states, actions, y):
        with tf.GradientTape() as t:
            current_loss = self.loss(y, self.predict(states, actions))
        grads = t.gradient(current_loss, self.weights)

        loss = lambda:tf.losses.mean_squared_error(y, self.predict(states, actions, axis=1))

        self.opt.minimize(loss, self.weights)
        return current_loss 
    
    def predict(self, states, actions, axis=1):
        return self.network(states, actions, axis)

    def predict_target(self, states, actions):
        return self.target_network(states,actions)

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
            


    # def train(self, states, actions, y):
    #     self.sess.run(self.optimize, \
    #                   feed_dict = {self.y:y, self.s:states, self.a:actions})

    # def predict(self, states, actions):
    #     self.sess.run(self.out, \
    #                   feed_dict = {self.s:states, self.actions:actions})

    # def predict_target(self, states, actions):
    #     self.sess.run(self.target_out, \
    #                   feed_dict = {self.target_s:states, self.target_actions:actions})

    # def update_target_network(self):
    #     for i in range(len(self.target_params)):
    #         self.target_params[i].assign(tf.multiply(self.params[i], self.tau) \
    #             + tf.multiply(self.target_params[i], 1. - self.tau))
