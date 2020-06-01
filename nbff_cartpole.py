import gym
import random
import torch
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from nbff_adam import *
import sys

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
opt_method = 'adam'
lr_decay   = 'f'
lr_choice  = '1'
# n          = 1
# n = int(sys.argv[1])
# exp_n = int(sys.argv[2])
n = 2
exp_n = 0

lr_choice = int(lr_choice)
if lr_decay == 'f':
    lrs = [1e-2, 1e-3, 1e-4]
elif lr_decay == 'd':
    lrs = [0.5, 1e-1, 1e-2, 1e-3]
lr = lrs[lr_choice]

train_episodes = 200
g              = 0.97 # Reward discount factor
batch_size     = 50

###############################################################################
# Training methods
###############################################################################
def train(n = n, opt_method = opt_method, lr = lr, lr_decay = lr_decay, train_episodes = train_episodes, batch_size = batch_size, Q = Net(), y = Net()):
    episode = 0
    e = 1.0
    memory = deque(maxlen = 10000)
    
    rwds = np.zeros(train_episodes)
    
    t = 0
    if opt_method == 'adam':
        ms = [torch.zeros(w.shape) for w in Q.parameters()]
        vs = [torch.zeros(w.shape) for w in Q.parameters()]
    
    for episode in range(train_episodes):
        nstep_buffer = deque(maxlen = n + 2)
        # nBFF requires n + 2 steps per gradient evaluation.
        # At the beginning of each episode, generate n steps to begin training.
        cur_s = env.reset()
        cur_a = int(e_greedy(Q(torch.Tensor(cur_s)), e))
        nstep_buffer.append((cur_s, cur_a, 0, False))
        
        total_rwd = 0
        
        for i in range(n + 1):
            cur_a = nstep_buffer[i][1]
            
            nxt_s, nxt_rwd, nxt_done, _ = env.step(cur_a)
            nxt_a                       = int(e_greedy(Q(torch.Tensor(nxt_s)), e))
            
            nstep_buffer.append((nxt_s, nxt_a, nxt_rwd, nxt_done))
        
        done = False
        while not done:
            memory.append(nstep_buffer.copy())
            
            sample_size = min(batch_size, len(memory))
            batch       = random.choices(memory, k=sample_size)
            
            t += 1
            if lr_decay == 'd':
                alpha = lr * (t ** (-0.75))
            elif lr_decay == 'f':
                alpha = lr
            
            if opt_method == 'sgd':
                sgd(batch, Q, alpha)
            elif opt_method == 'adam':
                adam(batch, Q, ms, vs, t, alpha, beta1=0.9, beta2=0.999)
            
            cur_a                       = nstep_buffer[-1][1]
            nxt_s, nxt_rwd, nxt_done, _ = env.step(cur_a)
            nxt_a                       = int(e_greedy(Q(torch.Tensor(nxt_s)), e))
            
            nstep_buffer.append((nxt_s, nxt_a, nxt_rwd, nxt_done))
            total_rwd += nstep_buffer[0][2]
            done       = nstep_buffer[0][3]
                
        print(f'Total reward for episode {episode + 1}: {total_rwd}')
        rwds[episode] = total_rwd
        e = max(0.1, 0.99 * e)
                    
    return Q, rwds
###############################################################################
# Run tests
###############################################################################
start = time.time()

env.seed(exp_n)
random.seed(exp_n)
np.random.seed(exp_n)
torch.manual_seed(exp_n)

Q, rwds = train()
torch.save(Q.state_dict(), f'cartpole/{n}bff_{opt_method}_q_{exp_n}')
env.close()
    
print(f'Total runtime: {time.time() - start}')

plt.style.use('ggplot')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(rwds)
plt.title(f'Reward vs. episode, {n}bff_{opt_method}_{exp_n}')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig(f'cartpole/{n}bff_{opt_method}_{exp_n}_plot.png')

with open(f'cartpole/{n}bff_{opt_method}_{exp_n}_rwd.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(rwds)