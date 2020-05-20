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


def find_bin(s, bins = 20):
    """
    Used for discretizing the state space.
    This allows us to use experience replay with uniform sampling.
    
    bins = number of bins per state feature
    """
    bin_index = '' * len(s)
    for i in range(len(s)):
        lo = env.observation_space.low[i]
        hi = env.observation_space.high[i]
        bin_size = (hi - lo) / bins
        bin_index += str(min(int((s[i] - lo) / bin_size), bins - 1))
    return bin_index


def update_memory(experience, memory, maxlen = 100, bins = 20):
    """
    Memory is a dict of max length bins^dim(state space).
    Each entry is a deque. This corresponds to a discretization of the state
    space.
    Each time we get a new experience from the trajectory, we add it to the
    bin for the corresponding discretized state.
    When we use experience replay, we can now sample nearly uniformly from the
    state space by first sampling uniformly from the different deques, then
    sampling randomly from each deque.
    
    experience = (cur_s, cur_a, nxt_s, new_s, nxt_rwd, nxt_done)
    """
    cur_s = experience[0]
    bin_index = find_bin(cur_s, bins)
    if bin_index not in memory:
        memory[bin_index] = deque(maxlen=maxlen)
    memory[bin_index].append(experience)


def unif_batch(memory, batch_size):
    batch = [None] * batch_size
    keys = list(memory)
    for i in range(batch_size):
        key = random.sample(keys, k=1)[0]
        unif_sample = memory[key]
        batch[i] = random.sample(unif_sample, k=1)[0]
    return batch
        
    
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
g             = 0.97 # Reward discount factor
e0            = 0.8
batch_size    = 50
rounds        = 1
max_episodes  = 1000
learning_rate = 0.001
maxlen        = 100
bins          = 20
unif          = True
experiment_n  = 5
# method      = sys.argv[1]
# opt_method  = sys.argv[2]

method = 'ds'
opt_method = 'sgd'

experiment_path = f'mountaincar_results/{experiment_n}'
if not os.path.isdir(experiment_path):
    os.mkdir(experiment_path)

results_path = f'mountaincar_results/{experiment_n}/{opt_method}'
if not os.path.isdir(results_path):
    os.mkdir(results_path)
    
###############################################################################
# Training methods
###############################################################################
def DS(max_episodes, learning_rate, batch_size, Q = Net(), opt_method = opt_method, e0 = e0):
    episode = 0
    e = e0
    buffer = deque(maxlen=maxlen)
    
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

        nxt_s, nxt_time_rwd, _, _ = env.step(cur_a)
        nxt_a                     = int(e_greedy(Q(torch.Tensor(nxt_s)), e))
        nxt_done                  = nxt_s[0] > 0.5
        if nxt_s[0] > max_pos:
            max_pos = nxt_s[0]
        
        total_time_rwd = 0
        total_rwd = 0
        
        cur_done = False
        while not cur_done:
            total_time_rwd += nxt_time_rwd
            
            ftr_s, ftr_time_rwd, _, _ = env.step(nxt_a)
            ftr_done                  = ftr_s[0] > 0.5
            new_s                     = nxt_s
            
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
            e = max(e * 0.99, 0.05)
            learning_rate *= 0.99
                
        print(f'Time for episode {episode + 1}: {-total_time_rwd}')
        rwds[episode] = total_rwd
        time_rwds[episode] = total_time_rwd
        max_positions[episode] = max_pos
                    
    return Q, rwds, time_rwds, max_positions


def BFF(max_episodes, learning_rate, batch_size, Q = Net(), opt_method = opt_method, e0 = e0):
    episode = 0
    e = e0
    buffer = deque(maxlen=maxlen)
    
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

        nxt_s, nxt_time_rwd, _, _ = env.step(cur_a)
        nxt_a                     = int(e_greedy(Q(torch.Tensor(nxt_s)), e))
        nxt_done                  = nxt_s[0] > 0.5
        if nxt_s[0] > max_pos:
            max_pos = nxt_s[0]
        
        total_time_rwd = 0
        total_rwd = 0
        
        cur_done = False
        while not cur_done:
            total_time_rwd += nxt_time_rwd
            
            ftr_s, ftr_time_rwd, _, _ = env.step(nxt_a)
            ftr_done                  = ftr_s[0] > 0.5
            new_s                     = cur_s + (ftr_s - nxt_s)
            
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
            e = max(e * 0.99, 0.05)
            learning_rate *= 0.99
                
        print(f'Time for episode {episode + 1}: {-total_time_rwd}')
        rwds[episode] = total_rwd
        time_rwds[episode] = total_time_rwd
        max_positions[episode] = max_pos
                    
    return Q, rwds, time_rwds, max_positions


def unif_training(method, max_episodes, learning_rate, batch_size, Q = Net(), opt_method = opt_method, e0 = e0):
    episode = 0
    e = e0
    memory = {}
    
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

        nxt_s, nxt_time_rwd, _, _ = env.step(cur_a)
        nxt_a                     = int(e_greedy(Q(torch.Tensor(nxt_s)), e))
        nxt_done                  = nxt_s[0] > 0.5
        if nxt_s[0] > max_pos:
            max_pos = nxt_s[0]
        
        total_time_rwd = 0
        total_rwd = 0
        
        cur_done = False
        while not cur_done:
            total_time_rwd += nxt_time_rwd
            
            ftr_s, ftr_time_rwd, _, _ = env.step(nxt_a)
            ftr_done                  = ftr_s[0] > 0.5
            if method == 'ds':
                new_s = nxt_s
            elif method == 'bff':
                new_s = cur_s + (ftr_s - nxt_s)
            
            nxt_rwd = mc_reward(nxt_s)
            # nxt_rwd = nxt_time_rwd + g * abs(ftr_s[1]) - abs(nxt_s[1]) # Reward based on potentials
            total_rwd += nxt_rwd
            
            env.render()
            
            experience = (cur_s, cur_a, nxt_s, new_s, nxt_rwd, nxt_done)
            update_memory(experience, memory, maxlen = maxlen, bins = bins)
            t += 1
            
            batch = unif_batch(memory, min(t, batch_size))
            
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
            e = max(e * 0.99, 0.05)
            learning_rate *= 0.99
                
        print(f'Time for episode {episode + 1}: {-total_time_rwd}')
        rwds[episode] = total_rwd
        time_rwds[episode] = total_time_rwd
        max_positions[episode] = max_pos
                    
    return Q, rwds, time_rwds, max_positions

###############################################################################
# Run tests
###############################################################################
rwds           = np.zeros((rounds, max_episodes))
time_rwds      = np.zeros((rounds, max_episodes))
max_positions  = np.zeros((rounds, max_episodes))

start = time.time()
for r in range(rounds):
    print(f'Running round {r}.')
    env.seed(r)
    random.seed(r)
    np.random.seed(r)
    torch.manual_seed(r)
    
    if unif:
        Q, rwds[r, :], time_rwds[r, :], max_positions[r, :] = unif_training(method, max_episodes, learning_rate, batch_size)
        
    else:
        if method == 'bff':
            Q, rwds[r, :], time_rwds[r, :], max_positions[r, :] = BFF(max_episodes, learning_rate, batch_size)
            torch.save(Q.state_dict(), f'mountaincar_results/{experiment_n}/{opt_method}/{method}_Q_{r}')
            env.close()
    
        if method == 'ds':
            Q, rwds[r, :], time_rwds[r, :], max_positions[r, :] = DS(max_episodes, learning_rate, batch_size)
            torch.save(Q.state_dict(), f'mountaincar_results/{experiment_n}/{opt_method}/{method}_Q_{r}')
            env.close()
        
    print(f'Total runtime: {time.time() - start}')

sem   = sem(ds_rwds, axis=0)
mean  = np.mean(ds_rwds, axis=0)
lo    = np.quantile(ds_rwds, 0.25, axis=0)
mid   = np.quantile(ds_rwds, 0.5,  axis=0)
hi    = np.quantile(ds_rwds, 0.75, axis=0)

x = range(max_episodes)

plt.style.use('ggplot')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, mean, label=f'{method}')
ax.fill_between(x, mean - sem, mean + sem, alpha = 0.3)
plt.title('Average episode reward')
plt.xlabel('Episode')
plt.ylabel('Average reward +/- SEM')
plt.legend()
# plt.show()
plt.savefig(f'mountaincar_results/{experiment_n}/{opt_method}/{method}_plot.png')


with open(f'mountaincar_results/{experiment_n}/{opt_method}/{method}_rwds.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in rwds:
        writer.writerow(row)

with open(f'mountaincar_results/{experiment_n}/{opt_method}/{method}_time_rwds.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in time_rwds:
        writer.writerow(row)

with open(f'mountaincar_results/{experiment_n}/{opt_method}/{method}_max_dist.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in max_positions:
        writer.writerow(row)