import gym
import random
import copy
import torch
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from adam import *
from scipy.stats import sem
import sys

# env.reset(): Resets the environment and returns an initial state
# env.step(action): Returns observation, reward, done, info
# observation: State vector (for cartpole, 4D)
# reward: Reward from last action
# done: True if episode is finished, otherwise false
# info: Extra stuff useful debugging, e.g. may contain probabilities of a given transition, etc. Don't use for learning.

env_name = sys.argv[1]
# env_name = "LunarLander-v2"
env = gym.make(env_name)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 100)
#         self.fc2 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(100, env.action_space.n)

    def forward(self, x):
        x = func.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x


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
    
    rwds = np.zeros(max_episodes)
    
    ms = [torch.zeros(w.shape) for w in Q.parameters()]
    vs = [torch.zeros(w.shape) for w in Q.parameters()]
    t  = 0
    
    for episode in range(max_episodes):
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
            
            # env.render()

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
        rwds[episode] = total_rwd
        e = max(0.1, 0.99 * e)
                    
    return Q, rwds


def adam_BFF(max_episodes, learning_rate, batch_size, Q = Net()):
    episode = 0
    e = 1.0
    buffer = deque(maxlen=10000)
    
    rwds = np.zeros(max_episodes)
    
    ms = [torch.zeros(w.shape) for w in Q.parameters()]
    vs = [torch.zeros(w.shape) for w in Q.parameters()]
    t  = 0
    
    for episode in range(max_episodes):
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
            
            # env.render()

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
        rwds[episode] = total_rwd
        e = max(0.1, 0.99 * e)
                    
    return Q, rwds


###############################################################################
# Run tests
###############################################################################
torch.manual_seed(0)
DS_Q = Net()
BFF_Q = copy.deepcopy(DS_Q)

rounds = 1
max_episodes = 200

ds_rwds  = np.zeros((rounds, max_episodes))
bff_rwds = np.zeros((rounds, max_episodes))

start = time.time()
for r in range(rounds):
    print(f'Running round {r}.')
    env.seed(r+5)
    np.random.seed(r+5)
    random.seed(r)
    DS_Q, ds_rwds[r, :] = my_adam_DS(max_episodes, 0.001, 50, DS_Q)
    env.close()
    
    env.seed(r)
    np.random.seed(r)
    random.seed(r)
    BFF_Q, bff_rwds[r, :] = adam_BFF(max_episodes, 0.001, 50, Q = BFF_Q)
    print(f'Total runtime: {time.time() - start}')


ds_sem   = sem(ds_rwds, axis=0)
ds_mean  = np.mean(ds_rwds, axis=0)
ds_lo    = np.quantile(ds_rwds, 0.25, axis=0)
ds_mid   = np.quantile(ds_rwds, 0.5,  axis=0)
ds_hi    = np.quantile(ds_rwds, 0.75, axis=0)

bff_sem   = sem(bff_rwds, axis=0)
bff_mean  = np.mean(bff_rwds, axis=0)
bff_lo    = np.quantile(bff_rwds, 0.25, axis=0)
bff_mid   = np.quantile(bff_rwds, 0.5,  axis=0)
bff_hi    = np.quantile(bff_rwds, 0.75, axis=0)

x = range(max_episodes)

plt.style.use('ggplot')

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, ds_mean, label='ds')
ax.fill_between(x, ds_mean - ds_sem, ds_mean + ds_sem, alpha = 0.3)

ax.plot(x, bff_mean, label='bff')
ax.fill_between(x, bff_mean - bff_sem, bff_mean + bff_sem, alpha = 0.3)

plt.title('Average episode reward')
plt.xlabel('Episode')
plt.ylabel('Average reward +/- SEM')
plt.legend()
plt.show()
plt.savefig(f'{env}_rwd.png')


with open(f'{env}_ds_rwds.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in ds_rwds:
        writer.writerow(row)

with open(f'{env}_bff_rwds.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in bff_rwds:
        writer.writerow(row)


# With all random seeds = 0, we get the following results:
# DS:
# First time reaching 200: Ep 178
# Hitting 200 almost all of the time by Ep 214
# BFF:
# First time reaching 200: Ep 77 (!)
# Hitting 200 almost all of the time by Ep 107 (!)