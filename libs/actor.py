import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input
from tensorflow.keras import optimizers
import numpy as np

class Actor(tf.keras.Model):
    
    def __init__(self, state_size, action_size, max_action, lr):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        

        W2 = np.random.uniform(-3e-3, 3e-3, (400, 300))
        b2 = np.random.uniform(-3e-3, 3e-3, 300)

        W3 = np.random.uniform(-3e-3, 3e-3, (300, self.action_size))
        b3 = np.random.uniform(-3e-3, 3e-3, self.action_size)

        self.max_action = max_action
        self.l1 = Dense(400)
        self.l2 = Dense(300, weights = [W2, b2])
        self.l3 = Dense(self.action_size, weights = [W3, b3])

    
        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros(shape=(1, self.state_size), dtype=np.float64)))

    def call(self, inputs):
        features = tf.nn.relu(self.l1(inputs))
        features = tf.nn.relu(self.l2(features))
        features = self.l3(features)
        action = self.max_action * tf.nn.tanh(features)
        return action

        

   
