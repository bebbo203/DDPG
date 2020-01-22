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
        model.add(Dense(400, activation="relu", input_dim=self.state_size))
        model.add(BatchNormalization())
        model.add(Dense(300, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(self.action_size, activation="tanh"))


        # new_w = np.random.uniform(low=-3e-3, high=3e-3, size=(400,300))
        # new_b = np.random.uniform(low=-3e-3, high=3e-3, size=(300))
        # model.layers[-3].set_weights([new_w, new_b])

        # new_w = np.random.uniform(low=-3e-3, high=3e-3, size=(300,1))
        # new_b = np.random.uniform(low=-3e-3, high=3e-3, size=(1))
        # model.layers[-1].set_weights([new_w, new_b])


        return model
