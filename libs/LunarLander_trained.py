from agent import Agent
import gym
import numpy as np
from plotter import *
import signal
from matplotlib import pyplot as plt
import os
import sys

actor_checkpoint_path = "training_actor/actor.ckpt"
critic_checkpoint_path = "training_critic/critic.ckpt"
actor_checkpoint_dir = os.path.dirname(actor_checkpoint_path)
critic_checkpoint_dir = os.path.dirname(critic_checkpoint_path)


env = gym.make("LunarLanderContinuous-v2")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

agent = Agent(state_dim, action_dim, action_bound, 128, 0.001, 0.0001, 0.99,  0.001)

agent.actor_network.load_weights(actor_checkpoint_path)
agent.actor_target_network.load_weights(actor_checkpoint_path)
agent.critic_network.load_weights(critic_checkpoint_path)
agent.critic_target_network.load_weights(critic_checkpoint_path)

while(True):
    s = env.reset()

    total_reward = 0
    t = False
    while(not t):
        env.render()

        a, _ = agent.act(s)
        s_1, r, t, _ = env.step(a)
        s = s_1
        total_reward += r

    print(total_reward)
