import gym
import random
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as func
from collections import deque
from adam import *

# env.reset(): Resets the environment and returns an initial state
# env.step(action): Returns observation, reward, done, info
# observation: State vector (for cartpole, 4D)
# reward: Reward from last action
# done: True if episode is finished, otherwise false
# info: Extra stuff useful debugging, e.g. may contain probabilities of a given transition, etc. Don't use for learning.

env_name = "CartPole-v0"
env = gym.make(env_name)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

###############################################################################
# Parameters
###############################################################################
g = 0.97 # Reward discount factor

###############################################################################
# Training methods
###############################################################################
def my_adam_DS(max_episodes, learning_rate, batch_size, Q = Net()):
    episode = 0
    e = 1.0
    buffer = deque(maxlen=10000)
    
    ms = [torch.zeros(w.shape) for w in Q.parameters()]
    vs = [torch.zeros(w.shape) for w in Q.parameters()]
    t  = 0
    
    while episode < max_episodes:
        cur_s = env.reset()
        cur_a = int(e_greedy(Q(torch.Tensor(cur_s)), e))

        nxt_s, nxt_rwd, nxt_done, _ = env.step(cur_a)
        nxt_a                       = int(e_greedy(Q(torch.Tensor(nxt_s)), e))

        total_rwd = 0
        
        cur_done = False
        while not cur_done:
            total_rwd += nxt_rwd
            
            ftr_s, ftr_rwd, ftr_done, _ = env.step(nxt_a)
            new_s                       = nxt_s
            
            env.render()

            buffer.append((cur_s, cur_a, nxt_s, new_s, nxt_rwd, nxt_done))
            
            sample_size = min(batch_size, len(buffer))
            batch       = random.choices(buffer, k=sample_size)
            
            t += 1                       
            adam(batch, Q, ms, vs, t)
            
            cur_s    = nxt_s
            cur_a    = nxt_a
            cur_done = nxt_done
            nxt_s    = ftr_s
            nxt_a    = int(e_greedy(Q(torch.Tensor(nxt_s)), e))
            nxt_done = ftr_done
            nxt_rwd  = ftr_rwd
                
        print(f'Total reward for episode {episode + 1}: {total_rwd}')
        episode += 1
        e = max(0.1, 0.99 * e)
                    
    return Q


def torch_adam_DS(max_episodes, learning_rate, batch_size, Q = Net()):
    episode = 0
    e = 1.0
    buffer = deque(maxlen=10000)
    optimizer = torch.optim.Adam(Q.parameters(), lr=learning_rate)
    
    while episode < max_episodes:
        cur_s = env.reset()
        cur_a = int(e_greedy(Q(torch.Tensor(cur_s)), e))

        nxt_s, nxt_rwd, nxt_done, _ = env.step(cur_a)
        nxt_a                       = int(e_greedy(Q(torch.Tensor(nxt_s)), e))

        total_rwd = 0
        
        cur_done = False
        while not cur_done:
            total_rwd += nxt_rwd
            
            ftr_s, ftr_rwd, ftr_done, _ = env.step(nxt_a)
            new_s                       = nxt_s
            
            env.render()

            buffer.append((cur_s, cur_a, nxt_s, new_s, nxt_rwd, nxt_done))
            
            sample_size = min(batch_size, len(buffer))
            batch       = random.choices(buffer, k=sample_size)
            
            loss = compute_batch_loss(batch, Q) / (2 * len(batch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cur_s    = nxt_s
            cur_a    = nxt_a
            cur_done = nxt_done
            nxt_s    = ftr_s
            nxt_a    = int(e_greedy(Q(torch.Tensor(nxt_s)), e))
            nxt_done = ftr_done
            nxt_rwd  = ftr_rwd
                
        print(f'Total reward for episode {episode + 1}: {total_rwd}')
        episode += 1
        e = max(0.1, 0.99 * e)
                    
    return Q


def adam_BFF(max_episodes, learning_rate, batch_size, Q = Net()):
    episode = 0
    e = 1.0
    buffer = deque(maxlen=10000)
    
    ms = [torch.zeros(w.shape) for w in Q.parameters()]
    vs = [torch.zeros(w.shape) for w in Q.parameters()]
    t  = 0
    
    while episode < max_episodes:
        cur_s = env.reset()
        cur_a = int(e_greedy(Q(torch.Tensor(cur_s)), e))

        nxt_s, nxt_rwd, nxt_done, _ = env.step(cur_a)
        nxt_a                       = int(e_greedy(Q(torch.Tensor(nxt_s)), e))

        total_rwd = 0
        
        cur_done = False
        while not cur_done:
            total_rwd += nxt_rwd
            
            ftr_s, ftr_rwd, ftr_done, _ = env.step(nxt_a)
            new_s                       = cur_s + (ftr_s - nxt_s)
            
            env.render()

            buffer.append((cur_s, cur_a, nxt_s, new_s, nxt_rwd, nxt_done))
            
            sample_size = min(batch_size, len(buffer))
            batch       = random.choices(buffer, k=sample_size)
            
            t += 1                       
            adam(batch, Q, ms, vs, t)
            
            cur_s    = nxt_s
            cur_a    = nxt_a
            cur_done = nxt_done
            nxt_s    = ftr_s
            nxt_a    = int(e_greedy(Q(torch.Tensor(nxt_s)), e))
            nxt_done = ftr_done
            nxt_rwd  = ftr_rwd
                
        print(f'Total reward for episode {episode + 1}: {total_rwd}')
        episode += 1
        e = max(0.1, 0.99 * e)
                    
    return Q


###############################################################################
# Run tests
###############################################################################
torch.manual_seed(0)
DS_Q = Net()
BFF_Q = copy.deepcopy(DS_Q)

#env.seed(0)
#np.random.seed(0)
#random.seed(0)
#DS_Q = my_adam_DS(300, 0.001, 50, DS_Q)
# First time reaching 200: Ep 178
# Hitting 200 almost all of the time by Ep 214

env.seed(0)
np.random.seed(0)
random.seed(0)
BFF_Q = adam_BFF(300, 0.001, 50, Q = BFF_Q)