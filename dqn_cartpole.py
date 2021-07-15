import gym
import random
import torch
import torch.optim as optim
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
# n          = int(sys.argv[5]) # {0, ..., 9}

lr_choice  = '1'
n          = 3

lr_choice = int(lr_choice)
lrs = [1e-2, 1e-3, 1e-4]
lr = lrs[lr_choice]

train_episodes = 200
g              = 0.97 # Reward discount factor
batch_size     = 50

###############################################################################
# Training methods
###############################################################################
def batch_loss(batch, Q, target_Q):
    loss = 0
    for sample in batch:
        cur_s, cur_a, nxt_s, nxt_rwd, nxt_done = sample
        if nxt_done:
            target = nxt_rwd
        else:
            target = nxt_rwd + g * torch.max(target_Q(torch.Tensor(nxt_s)))
        loss += (target - Q(torch.Tensor(cur_s))[cur_a]) ** 2
        
    loss /= (2 * len(batch))
    return loss




def train(target_update = 10, lr = lr, train_episodes = train_episodes, batch_size = batch_size, Q = Net(), target_Q = Net()):
    episode = 0
    e = 1.0
    buffer = deque(maxlen = 10000)
    
    rwds = np.zeros(train_episodes)
    
    optimizer = optim.Adam(Q.parameters(), lr=lr)
    
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

            buffer.append((cur_s, cur_a, nxt_s, nxt_rwd, nxt_done))
            
            sample_size = min(batch_size, len(buffer))
            batch       = random.choices(buffer, k=sample_size)
            
            loss = batch_loss(batch, Q, target_Q)
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
        rwds[episode] = total_rwd
        e = max(0.1, 0.99 * e)
        if (episode + 1) % target_update == 0:
            target_Q.load_state_dict(Q.state_dict())
                    
    return Q, rwds
###############################################################################
# Run tests
###############################################################################
start = time.time()

env.seed(n)
random.seed(n)
np.random.seed(n)
torch.manual_seed(n)

Q, rwds = train()
torch.save(Q.state_dict(), f'cartpole/random/dqn_{n}')
env.close()
    
print(f'Total runtime: {time.time() - start}')

plt.style.use('ggplot')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(rwds)
plt.title(f'Reward vs. episode, dqn_{n}')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig(f'cartpole/random/dqn_{n}_plot.png')

with open(f'cartpole/random/dqn_{n}_rwd.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(rwds)