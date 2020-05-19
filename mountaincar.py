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
import os

# env.reset(): Resets the environment and returns an initial state
# env.step(action): Returns observation, reward, done, info
# observation: State vector (for cartpole, 4D)
# reward: Reward from last action
# done: True if episode is finished, otherwise false
# info: Extra stuff useful debugging, e.g. may contain probabilities of a given transition, etc. Don't use for learning.

env_name = "MountainCar-v0"
env = gym.make(env_name)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

###############################################################################
# MountainCar-specific functions
###############################################################################
def mc_reward(s):
    rwd = s[0] + 0.5 # reward for being farther to the right
    if s[0] >= 0.5:  # extra reward for successful completion
        rwd += 1
    return rwd
    
###############################################################################
# NN architecture
###############################################################################
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 100, bias=False)
#         self.fc2 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(100, env.action_space.n, bias=False)

    def forward(self, x):
        # x = func.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

###############################################################################
# Parameters
###############################################################################
g = 0.97 # Reward discount factor
batch_size = 1
rounds = 1
max_episodes = 10
experiment_n = 4
method = sys.argv[1]
opt_method = sys.argv[2]

# method = 'ds'
# opt_method = 'sgd'

experiment_path = f'mountaincar_results/{experiment_n}'
if not os.path.isdir(experiment_path):
    os.mkdir(experiment_path)

results_path = f'mountaincar_results/{experiment_n}/{opt_method}'
if not os.path.isdir(results_path):
    os.mkdir(results_path)
    
###############################################################################
# Training methods
###############################################################################
def DS(max_episodes, learning_rate, batch_size, Q = Net(), opt_method = opt_method):
    episode = 0
    e = 0.5
    buffer = deque(maxlen=1)
    
    rwds          = np.zeros(max_episodes)
    time_rwds     = np.zeros(max_episodes)
    max_positions = np.zeros(max_episodes)
    
    ms = [torch.zeros(w.shape) for w in Q.parameters()]
    vs = [torch.zeros(w.shape) for w in Q.parameters()]
    t  = 0
    
    for episode in range(max_episodes):
        cur_s   = env.reset()
        cur_a   = int(e_greedy(Q(torch.Tensor(cur_s)), e))
        max_pos = cur_s[0]

        nxt_s, nxt_time_rwd, nxt_done, _ = env.step(cur_a)
        nxt_a                            = int(e_greedy(Q(torch.Tensor(nxt_s)), e))
        if nxt_s[0] > max_pos:
            max_pos = nxt_s[0]
        
        total_time_rwd = 0
        total_rwd = 0
        
        cur_done = False
        while not cur_done:
            total_time_rwd += nxt_time_rwd
            
            ftr_s, ftr_time_rwd, ftr_done, _ = env.step(nxt_a)
            new_s                            = nxt_s
            
            nxt_rwd = mc_reward(nxt_s)
            # nxt_rwd = nxt_time_rwd + g * abs(ftr_s[1]) - abs(nxt_s[1]) # Reward based on potentials
            total_rwd += nxt_rwd
            
            # env.render()

            buffer.append((cur_s, cur_a, nxt_s, new_s, nxt_rwd, nxt_done))
            
            sample_size = min(batch_size, len(buffer))
            batch       = random.choices(buffer, k=sample_size)
            
            t += 1
            if opt_method == 'adam':
                adam(batch, Q, ms, vs, t, lr = learning_rate)
            elif opt_method == 'sgd':
                sgd(batch, Q, learning_rate)
            
            cur_s         = nxt_s
            cur_a         = nxt_a
            cur_done      = nxt_done
            nxt_s         = ftr_s
            nxt_a         = int(e_greedy(Q(torch.Tensor(nxt_s)), e))
            nxt_done      = ftr_done
            nxt_time_rwd  = ftr_time_rwd
            
            if nxt_s[0] > max_pos:
                max_pos = nxt_s[0]
            
        if cur_s[0] >= 0.5:
            e = max(e * 0.95, 0.05)
            learning_rate *= 0.9
                
        print(f'Total reward for episode {episode + 1}: {total_rwd}')
        rwds[episode] = total_rwd
        time_rwds[episode] = total_time_rwd
        max_positions[episode] = max_pos
                    
    return Q, rwds, time_rwds, max_positions


def BFF(max_episodes, learning_rate, batch_size, Q = Net(), opt_method = opt_method):
    episode = 0
    e = 0.5
    buffer = deque(maxlen=1)
    
    rwds          = np.zeros(max_episodes)
    time_rwds     = np.zeros(max_episodes)
    max_positions = np.zeros(max_episodes)
    
    ms = [torch.zeros(w.shape) for w in Q.parameters()]
    vs = [torch.zeros(w.shape) for w in Q.parameters()]
    t  = 0
    
    for episode in range(max_episodes):
        cur_s   = env.reset()
        cur_a   = int(e_greedy(Q(torch.Tensor(cur_s)), e))
        max_pos = cur_s[0]

        nxt_s, nxt_time_rwd, nxt_done, _ = env.step(cur_a)
        nxt_a                            = int(e_greedy(Q(torch.Tensor(nxt_s)), e))
        if nxt_s[0] > max_pos:
            max_pos = nxt_s[0]
        
        total_time_rwd = 0
        total_rwd = 0
        
        cur_done = False
        while not cur_done:
            total_time_rwd += nxt_time_rwd
            
            ftr_s, ftr_time_rwd, ftr_done, _ = env.step(nxt_a)
            new_s                            = cur_s + (ftr_s - nxt_s)
            
            nxt_rwd = mc_reward(nxt_s)
            # nxt_rwd = nxt_time_rwd + g * abs(ftr_s[1]) - abs(nxt_s[1]) # Reward based on potentials
            total_rwd += nxt_rwd
            
            # env.render()

            buffer.append((cur_s, cur_a, nxt_s, new_s, nxt_rwd, nxt_done))
            
            sample_size = min(batch_size, len(buffer))
            batch       = random.choices(buffer, k=sample_size)
            
            t += 1
            if opt_method == 'adam':
                adam(batch, Q, ms, vs, t, lr = learning_rate)
            elif opt_method == 'sgd':
                sgd(batch, Q, learning_rate)
            
            cur_s         = nxt_s
            cur_a         = nxt_a
            cur_done      = nxt_done
            nxt_s         = ftr_s
            nxt_a         = int(e_greedy(Q(torch.Tensor(nxt_s)), e))
            nxt_done      = ftr_done
            nxt_time_rwd  = ftr_time_rwd
            
            if nxt_s[0] > max_pos:
                max_pos = nxt_s[0]
            
        if cur_s[0] >= 0.5:
            e = max(e * 0.95, 0.05)
            learning_rate *= 0.9
                
        print(f'Total reward for episode {episode + 1}: {total_rwd}')
        rwds[episode] = total_rwd
        time_rwds[episode] = total_time_rwd
        max_positions[episode] = max_pos
                    
    return Q, rwds, time_rwds, max_positions


###############################################################################
# Run tests
###############################################################################
torch.manual_seed(0)
DS_Q = Net()
BFF_Q = copy.deepcopy(DS_Q)

ds_rwds  = np.zeros((rounds, max_episodes))
bff_rwds = np.zeros((rounds, max_episodes))

ds_time_rwds  = np.zeros((rounds, max_episodes))
bff_time_rwds = np.zeros((rounds, max_episodes))

ds_max_positions  = np.zeros((rounds, max_episodes))
bff_max_positions = np.zeros((rounds, max_episodes))

start = time.time()
for r in range(rounds):
    print(f'Running round {r}.')
    
    if method == 'bff':
        env.seed(r)
        np.random.seed(r)
        random.seed(r)
        BFF_Q, bff_rwds[r, :], bff_time_rwds[r, :], bff_max_positions[r, :] = BFF(max_episodes, 0.001, batch_size, Q = BFF_Q)
        torch.save(BFF_Q.state_dict(), f'mountaincar_results/{experiment_n}/{opt_method}/{method}_Q_{r}')
        env.close()
    
    if method == 'ds':
        env.seed(r)
        np.random.seed(r)
        random.seed(r)
        DS_Q, ds_rwds[r, :], ds_time_rwds[r, :], ds_max_positions[r, :] = DS(max_episodes, 0.001, batch_size, Q = DS_Q)
        torch.save(BFF_Q.state_dict(), f'mountaincar_results/{experiment_n}/{opt_method}/{method}_Q_{r}')
        env.close()
        
    print(f'Total runtime: {time.time() - start}')

if method == 'ds':
    ds_sem   = sem(ds_rwds, axis=0)
    ds_mean  = np.mean(ds_rwds, axis=0)
    ds_lo    = np.quantile(ds_rwds, 0.25, axis=0)
    ds_mid   = np.quantile(ds_rwds, 0.5,  axis=0)
    ds_hi    = np.quantile(ds_rwds, 0.75, axis=0)

if method == 'bff':
    bff_sem   = sem(bff_rwds, axis=0)
    bff_mean  = np.mean(bff_rwds, axis=0)
    bff_lo    = np.quantile(bff_rwds, 0.25, axis=0)
    bff_mid   = np.quantile(bff_rwds, 0.5,  axis=0)
    bff_hi    = np.quantile(bff_rwds, 0.75, axis=0)

x = range(max_episodes)

plt.style.use('ggplot')

fig = plt.figure()
ax = fig.add_subplot(111)

if method == 'ds':
    ax.plot(x, ds_mean, label='ds')
    ax.fill_between(x, ds_mean - ds_sem, ds_mean + ds_sem, alpha = 0.3)

if method == 'bff':
    ax.plot(x, bff_mean, label='bff')
    ax.fill_between(x, bff_mean - bff_sem, bff_mean + bff_sem, alpha = 0.3)

plt.title('Average episode reward')
plt.xlabel('Episode')
plt.ylabel('Average reward +/- SEM')
plt.legend()
# plt.show()
plt.savefig(f'mountaincar_results/{experiment_n}/{opt_method}/{method}_plot.png')


with open(f'mountaincar_results/{experiment_n}/{opt_method}/{method}_rwds.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if method == 'ds':
        for row in ds_rwds:
            writer.writerow(row)
            
    elif method == 'bff':
        for row in bff_rwds:
            writer.writerow(row)

with open(f'mountaincar_results/{experiment_n}/{opt_method}/{method}_time_rwds.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if method == 'ds':
        for row in ds_time_rwds:
            writer.writerow(row)
            
    elif method == 'bff':
        for row in bff_time_rwds:
            writer.writerow(row)

with open(f'mountaincar_results/{experiment_n}/{opt_method}/{method}_max_dist.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if method == 'ds':
        for row in ds_max_positions:
            writer.writerow(row)
            
    elif method == 'bff':
        for row in bff_max_positions:
            writer.writerow(row)