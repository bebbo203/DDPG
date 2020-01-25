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
        
        self.max_action = max_action
        self.l1 = Dense(64)
        #self.l2 = Dense(10)
        self.l3 = Dense(self.action_size)
        
        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros(shape=(1, self.state_size), dtype=np.float64)))

    def call(self, inputs):
        features = tf.nn.relu(self.l1(inputs))
        #features = tf.nn.relu(self.l2(features))
        features = self.l3(features)
        action = self.max_action * tf.nn.tanh(features)
        return action

        

   
