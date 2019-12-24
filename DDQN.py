import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
import tensorflow as tf
import gym
from random import sample



def create_net_Q(input__dim, output_dim):
    model = Sequential()
    model.add(Dense(4, input_dim=input__dim, activation="softmax"))
    model.add(Dense(output_dim))

    model.compile(loss=losses.mean_squared_error, optimizer="adam")

    return model

def create_net_Mu(input__dim, output_dim):
    model = Sequential()
    model.add(Dense(4, input_dim=input__dim, activation="softmax"))
    model.add(Dense(output_dim))
    model.compile(loss=losses.mean_squared_error, optimizer='rmsprop')

    return model

def memorize(l, elem, batch_size):
    l.append(elem)
    if(len(l) > batch_size):
        del l[0]
        





state_dim = 2
action_dim = 1
EPISODES = 5
MINIBATCH = 64
MEMORY = 1000
gamma = 0.01


Q_net = create_net_Q(state_dim+action_dim, 1)
Q_target_net = create_net_Q(state_dim+action_dim, 1)
mu_net = create_net_Mu(state_dim, action_dim)
mu_target_net = create_net_Mu(state_dim, action_dim)
replay_buffer = []
env = gym.make("MountainCarContinuous-v0")


max_r = -999999999

for n_episode in range(EPISODES):
    s = env.reset()
    #env.render()
    d = False
    while(not d):
        a = mu_net.predict(np.array([s]), batch_size=1)
        #print(a)
        #env.render()
        s_next, r, d, _ =  env.step(a)
        s_next = np.array(s_next).flatten()
        memorize(replay_buffer, [s, a[0], np.array(s_next).flatten() , r, d], MEMORY)

        s = s_next
        
        if(r > max_r):
            print(r)
            max_r = r

        if(len(replay_buffer) >= MINIBATCH):
            minibatch = sample(replay_buffer, MINIBATCH)
            y = []
            actions = []
            Q_values = []
            for s, a, s_next, r, d in minibatch:
                next_action = mu_target_net.predict(np.array([s_next]))[0]
                if(not d):
                    target = r + gamma * Q_target_net.predict(np.array([np.concatenate((s_next, next_action))]))
                else:
                    target = r
                Q_net.fit(np.array([np.concatenate((s_next, next_action))]), target, verbose=0, steps_per_epoch = 1)
                mu_net.fit(np.array([s]), a, verbose=0, steps_per_epoch = 1)

        
        rho = 0.05
        for net, target in zip(Q_net.layers, Q_target_net.layers):
            net_w = net.get_weights()
            net_w[0]*=rho
            target_w = target.get_weights()
            target_w[0]*=(1-rho)

            target_w[0]+=net_w[0]
            target.set_weights(target_w)

        for net, target in zip(mu_net.layers, mu_target_net.layers):
            net_w = net.get_weights()
            net_w[0]*=rho
            target_w = target.get_weights()
            target_w[0]*=(1-rho)

            target_w[0]+=net_w[0]
            target.set_weights(target_w)






