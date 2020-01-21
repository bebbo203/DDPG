import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input
from tensorflow.keras import optimizers
import numpy as np

class Actor(object):
    def __init__(self, state_size, action_size, lr):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.model = self.create_actor_network()

    def create_actor_network(self):
        model = tf.keras.models.Sequential()
        model.add(Dense(200, activation="relu", input_dim=self.state_size))
        model.add(BatchNormalization())
        model.add(Dense(200, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(self.action_size, activation="tanh"))
        return model
