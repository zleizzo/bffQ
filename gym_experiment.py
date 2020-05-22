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

# env_name = sys.argv[1]
env_name = "LunarLander-v2"
env = gym.make(env_name)

###############################################################################
# Parameters
###############################################################################
experiment_n   = 2

new_method     = sys.argv[1]
opt_method     = sys.argv[2]
# new_method     = 'ds'
# opt_method     = 'adam'

train_episodes = 1000
rounds         = 1
g              = 0.99
lr             = 1e-4
beta1          = 0.9
beta2          = 0.98

e_start        = 0.5
e_decay        = 0.99
e_min          = 0
memory_size    = 65536
batch_size     = 32
training_start = 64

fc1_size       = 256
fc2_size       = 128

print(f'env_name       = {env_name}')
print(f'experiment_n   = {experiment_n}')
print(f'new_method     = {new_method}')
print(f'opt_method     = {opt_method}')
print(f'train_episodes = {train_episodes}')
print(f'rounds         = {rounds}')
print(f'g              = {g}')
print(f'lr             = {lr}')
print(f'e_start        = {e_start}')
print(f'e_decay        = {e_decay}')
print(f'e_min          = {e_min}')
print(f'memory_size    = {memory_size}')
print(f'batch_size     = {batch_size}')
print(f'fc1_size       = {fc1_size}')
print(f'fc2_size       = {fc2_size}')


experiment_path = f'lunarlander_results/{experiment_n}'
if not os.path.isdir(experiment_path):
    os.mkdir(experiment_path)

results_path = f'lunarlander_results/{experiment_n}/{opt_method}'
if not os.path.isdir(results_path):
    os.mkdir(results_path)

###############################################################################
# NN architecture
###############################################################################
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, env.action_space.n)

    def forward(self, x):
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x

###############################################################################
# Training methods
###############################################################################
def my_adam_DS(train_episodes, learning_rate, batch_size, Q = Net()):
    episode = 0
    e = e_start
    buffer = deque(maxlen=memory_size)
    
    rwds = np.zeros(train_episodes)
    
    ms = [torch.zeros(w.shape) for w in Q.parameters()]
    vs = [torch.zeros(w.shape) for w in Q.parameters()]
    t  = 0
    
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
def adam_BFF(train_episodes, learning_rate, batch_size, Q = Net()):
    episode = 0
    e = 1.0
    buffer = deque(maxlen=10000)
    
    rwds = np.zeros(train_episodes)
    
    ms = [torch.zeros(w.shape) for w in Q.parameters()]
    vs = [torch.zeros(w.shape) for w in Q.parameters()]
    t  = 0
    
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

def train(train_episodes, learning_rate, batch_size, Q = Net(), new_method = new_method, opt_method = opt_method):
    memory = deque(maxlen = memory_size)
    rwds   = np.zeros(train_episodes)
    
    episode = 0
    e       = e_start
    
    if opt_method == 'adam':
        ms = [torch.zeros(w.shape) for w in Q.parameters()]
        vs = [torch.zeros(w.shape) for w in Q.parameters()]
        t  = 0
    
    for episode in range(train_episodes):
        total_rwd = 0
        
        cur_s    = env.reset()
        cur_a    = int(e_greedy(Q(torch.Tensor(cur_s)), e))
        cur_done = False

        nxt_s, nxt_rwd, nxt_done, _ = env.step(cur_a)
        nxt_a                            = int(e_greedy(Q(torch.Tensor(nxt_s)), e))
        
        while not cur_done:
            total_rwd += nxt_rwd
            
            ftr_s, ftr_rwd, ftr_done, _ = env.step(nxt_a)
            
            if new_method == 'ds':
                new_s = nxt_s
            elif new_method == 'bff':
                new_s = cur_s + (ftr_s - nxt_s)
            
            # env.render()
            
            experience = (cur_s, cur_a, nxt_s, new_s, nxt_rwd, nxt_done)
            memory.append(experience)
            
            if len(memory) >= training_start:
                sample_size = min(batch_size, len(memory))
                batch       = random.choices(memory, k = sample_size)
                
                if opt_method == 'adam':
                    t += 1
                    adam(batch, Q, ms, vs, t, lr = learning_rate, beta1 = beta1, beta2 = beta2)
                elif opt_method == 'sgd':
                    sgd(batch, Q, learning_rate)
            
            cur_s    = nxt_s
            cur_a    = nxt_a
            cur_done = nxt_done
            
            nxt_s    = ftr_s
            nxt_a    = int(e_greedy(Q(torch.Tensor(nxt_s)), e))
            nxt_done = ftr_done
            nxt_rwd  = ftr_rwd
            
            e = max(e * e_decay, e_min)
                
        print(f'Time for episode {episode + 1}: {total_rwd}')
        rwds[episode] = total_rwd
                    
    return Q, rwds
###############################################################################
# Run tests
###############################################################################
rwds           = np.zeros((rounds, train_episodes))
time_rwds      = np.zeros((rounds, train_episodes))
max_positions  = np.zeros((rounds, train_episodes))

start = time.time()
for r in range(rounds):
    print(f'Running round {r}.')
    env.seed(r)
    random.seed(r)
    np.random.seed(r)
    torch.manual_seed(r)
    
    Q, rwds[r, :] = train(train_episodes, lr, batch_size)
    torch.save(Q.state_dict(), f'lunarlander_results/{experiment_n}/{opt_method}/{new_method}_Q_{r}')
    env.close()
        
    print(f'Total runtime: {time.time() - start}')

sem   = sem(rwds, axis=0)
mean  = np.mean(rwds, axis=0)
lo    = np.quantile(rwds, 0.25, axis=0)
mid   = np.quantile(rwds, 0.5,  axis=0)
hi    = np.quantile(rwds, 0.75, axis=0)

x = range(train_episodes)

plt.style.use('ggplot')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, mean, label=f'{new_method}')
ax.fill_between(x, mean - sem, mean + sem, alpha = 0.3)
plt.title('Average episode reward')
plt.xlabel('Episode')
plt.ylabel('Average reward +/- SEM')
plt.legend()
# plt.show()
plt.savefig(f'lunarlander_results/{experiment_n}/{opt_method}/{new_method}_plot.png')


with open(f'lunarlander_results/{experiment_n}/{opt_method}/{new_method}_rwds.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in rwds:
        writer.writerow(row)