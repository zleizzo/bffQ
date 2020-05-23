import gym
import random
import torch
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from adam import *
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
# new_method = sys.argv[1] # {ds, bff, pd}
# opt_method = sys.argv[2] # {sgd, adam}
# lr_decay   = sys.argv[3] # {f, d}
# lr_choice  = sys.argv[4] # {0, 1, 2}
# n          = sys.argv[5] # {0, ..., 9}

new_method = 'bff'
opt_method = 'adam'
lr_decay   = 'f'
lr_choice  = '1'

lr_choice = int(lr_choice)
if lr_decay == 'f':
    lrs = [1e-2, 1e-3, 1e-4]
elif lr_decay == 'd':
    lrs = [1e-1, 1e-2, 1e-3]
lr = lrs[lr_choice]

train_episodes = 1000
g              = 0.97 # Reward discount factor
batch_size     = 50

###############################################################################
# Training methods
###############################################################################
def train(new_method = new_method, opt_method = opt_method, lr = lr, lr_decay = lr_decay, train_episodes = train_episodes, batch_size = batch_size, Q = Net(), y = Net()):
    episode = 0
    e = 1.0
    buffer = deque(maxlen = 10000)
    
    rwds = np.zeros(train_episodes)
    
    t = 0
    if opt_method == 'adam':
        ms = [torch.zeros(w.shape) for w in Q.parameters()]
        vs = [torch.zeros(w.shape) for w in Q.parameters()]
    
    for episode in range(train_episodes):
        cur_s = env.reset()
        cur_a = int(e_greedy(Q(torch.Tensor(cur_s)), e))

        nxt_s, nxt_rwd, nxt_done, _ = env.step(cur_a)
        nxt_a                       = int(e_greedy(Q(torch.Tensor(nxt_s)), e))

        total_rwd = 0
        
        cur_done = False
        while not cur_done:
            total_rwd += nxt_rwd
            
            ftr_s, ftr_rwd, ftr_done, _ = env.step(nxt_a)
            
            if new_method == 'bff':
                new_s = cur_s + (ftr_s - nxt_s)
            else:
                new_s = nxt_s

            buffer.append((cur_s, cur_a, nxt_s, new_s, nxt_rwd, nxt_done))
            
            sample_size = min(batch_size, len(buffer))
            batch       = random.choices(buffer, k=sample_size)
            
            t += 1
            if lr_decay == 'd':
                alpha = lr * (t ** (-0.75))
                beta  = lr * (t ** (-0.75))
                eta   = lr * (t ** (-0.5))
            elif lr_decay == 'f':
                alpha = lr
                beta  = lr
                eta   = lr
            
            if new_method == 'pd':
                pd_step(batch, Q, y, beta, eta)
            elif opt_method == 'sgd':
                sgd(batch, Q, alpha)
            elif opt_method == 'adam':
                adam(batch, Q, ms, vs, t, alpha, beta1=0.9, beta2=0.999)
            
            cur_s    = nxt_s
            cur_a    = nxt_a
            cur_done = nxt_done
            nxt_s    = ftr_s
            nxt_a    = int(e_greedy(Q(torch.Tensor(nxt_s)), e))
            nxt_done = ftr_done
            nxt_rwd  = ftr_rwd
                
        print(f'Total reward for episode {episode + 1}: {total_rwd}')
        rwds[episode] = total_rwd
        e = max(0.1, 0.99 * e)
                    
    return Q, rwds
###############################################################################
# Run tests
###############################################################################
start = time.time()

env.seed(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

Q, rwds = train()
torch.save(Q.state_dict(), f'cartpole/{new_method}_{opt_method}_{lr_decay}_{lr_choice}_q')
env.close()
    
print(f'Total runtime: {time.time() - start}')

plt.style.use('ggplot')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(rwds)
plt.title(f'Reward vs. episode, {new_method}_{opt_method}_{lr_decay}_{lr_choice}')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig(f'cartpole/{new_method}_{opt_method}_{lr_decay}_{lr_choice}_plot.png')

with open(f'cartpole/{new_method}_{opt_method}_{lr_decay}_{lr_choice}_rwd.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(rwds)