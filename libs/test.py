from actor import Actor
from critic import Critic
from replayBuffer import ReplayBuffer
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import gym

state_dim = 3
action_dim = 1
EPISODES = 100
MINIBATCH_SIZE = 32
BUFFER_SIZE = 500
T_MAX = 50
G = 0.01



env = gym.make("Pendulum-v0")
actor = Actor(state_dim, action_dim, 0, MINIBATCH_SIZE, 0.0001, 0.1)
critic = Critic(state_dim, action_dim, 0, MINIBATCH_SIZE, 0.0001, 0.1)
replay = ReplayBuffer(BUFFER_SIZE)


for n_episode in range(EPISODES):
    s = env.reset()
    s = [s.astype(dtype="float32")]

    d = False
    t = 0
    total_r = 0
    for t in range(T_MAX):
        env.render()
        a = actor.predict(s)
        print(a)
        #print(a)
        s_next, r, d, _ = env.step(a)
        r = r
        s_next = [s_next.flatten().astype(dtype="float32")]
        total_r += r
        replay.add(s, a, r, s_next, d)

        if(replay.size() >= MINIBATCH_SIZE):

            s_batch, a_batch, r_batch, s_1_batch, d_batch = \
                replay.sample_batch(MINIBATCH_SIZE)

            #Critic Train
            y = []
            for i in range(MINIBATCH_SIZE):
                predicted_action = actor.predict_target(s_1_batch[i])
                predicted_Q = critic.predict_target(s_1_batch[1], predicted_action)

               
                if(d_batch[i] == False):
                    y.append(r_batch[i] + G*predicted_Q)
                else:
                    y.append(r_batch[i])

            loss = critic.train_batch(s_batch, a_batch, y)
            #Actor Train
            critic_grads = critic.get_critic_gradient(s_batch, a_batch)

            
            #for i in range(MINIBATCH_SIZE):
            #    critic_grads.append(critic.get_critic_gradient(s_batch[i], a_batch[i]))
            
            actor.train_batch(s_batch, critic_grads)

            actor.update_target_network()
            critic.update_target_network()
    print(total_r)  
            
        
         
        
        
        
