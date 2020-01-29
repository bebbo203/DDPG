import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input, Concatenate
from tensorflow.keras import optimizers
import numpy as np

class Critic(tf.keras.Model):
    def __init__(self, state_size, action_size, lr):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr



        W2 = np.random.uniform(-3e-3, 3e-3, (400, 300))
        b2 = np.random.uniform(-3e-3, 3e-3, 300)

        W3 = np.random.uniform(-3e-3, 3e-3, (300, self.action_size))
        b3 = np.random.uniform(-3e-3, 3e-3, self.action_size)

        


        self.l1 = Dense(400)
        self.l2 = Dense(300, weights = [W2, b2])
        self.l3 = Dense(self.action_size, weights = [W3, b3])
        self.state_shape=(1, state_size)

        dummy_state = tf.constant(
            np.zeros(shape=(1, self.state_size), dtype=np.float64))
        dummy_action = tf.constant(
            np.zeros(shape=[1, self.action_size], dtype=np.float64))
        with tf.device("/gpu:0"):
            self([dummy_state, dummy_action])


    def call(self, inputs):
        states, actions = inputs
        features = tf.concat([states, actions], axis=1)
        features = tf.nn.relu(self.l1(features))
        features = tf.nn.relu(self.l2(features))
        features = self.l3(features)
        return tf.cast(features, dtype=tf.float64)

    