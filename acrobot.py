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

method = sys.argv[1]

env_name = "Acrobot-v1"
env = gym.make(env_name)

# The state consists of the sin() and cos() of the two rotational joint
# angles and the joint angular velocities :
# [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
# For the first link, an angle of 0 corresponds to the link pointing downwards.
# The angle of the second link is relative to the angle of the first link.
# An angle of 0 corresponds to having the same angle between the two links.
# A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
# **ACTIONS:**
# The action is either applying +1, 0 or -1 torque on the joint between
# the two pendulum links.
#
# The end of the Acrobot is at cos(theta1) + cos(theta1 + theta2).
# (Positive points down.)
# Note that cos(theta1 + theta2) = cos(theta1)cos(theta2) - sin(theta1)sin(theta2)
# Thus the tip position given state s is
# tip = s[0] + (s[0] * s[2] - s[1] * s[3])
# Success is reached when tip <= -1.

def tip(s):
    return s[0] + (s[0] * s[2] - s[1] * s[3])

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 200)
#         self.fc2 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(200, env.action_space.n)

    def forward(self, x):
        x = func.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x

###############################################################################
# Parameters
###############################################################################
g            = 0.97 # Reward discount factor
c            = 1    # Reward imbalance for right vs. left
e_decay      = 0.99
rounds       = 1
max_episodes = 1
experiment_n = 2

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
        while tip(cur_s) > -1:
            total_rwd += nxt_rwd
            
            ftr_s, ftr_rwd, ftr_done, _ = env.step(nxt_a)
            new_s                       = nxt_s
            
            # env.render()

            buffer.append((cur_s, cur_a, nxt_s, new_s, nxt_rwd, nxt_done))
            
            sample_size = min(batch_size, len(buffer))
            batch       = random.choices(buffer, k=sample_size)
            
            t += 1                       
            adam(batch, Q, ms, vs, t)
            
            cur_s         = nxt_s
            cur_a         = nxt_a
            cur_done      = nxt_done
            nxt_s         = ftr_s
            nxt_a         = int(e_greedy(Q(torch.Tensor(nxt_s)), e))
            nxt_done      = ftr_done
            nxt_rwd       = ftr_rwd
                
        print(f'Total reward for episode {episode + 1}: {total_rwd}')
        rwds[episode] = total_rwd
        e = max(0.1, e_decay * e)
                    
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
        nxt_a                            = int(e_greedy(Q(torch.Tensor(nxt_s)), e))

        total_time_rwd = 0
        total_rwd = 0
        
        cur_done = False
        while tip(cur_s) > -1:
            total_rwd += nxt_rwd
            
            ftr_s, ftr_rwd, ftr_done, _ = env.step(nxt_a)
            new_s                       = cur_s + (ftr_s - nxt_s)
            
            # env.render()

            buffer.append((cur_s, cur_a, nxt_s, new_s, nxt_rwd, nxt_done))
            
            sample_size = min(batch_size, len(buffer))
            batch       = random.choices(buffer, k=sample_size)
            
            t += 1                       
            adam(batch, Q, ms, vs, t)
            
            cur_s         = nxt_s
            cur_a         = nxt_a
            cur_done      = nxt_done
            nxt_s         = ftr_s
            nxt_a         = int(e_greedy(Q(torch.Tensor(nxt_s)), e))
            nxt_done      = ftr_done
            nxt_rwd       = ftr_rwd
                            
        print(f'Total reward for episode {episode + 1}: {total_rwd}')
        rwds[episode] = total_rwd
        e = max(0.1, e_decay * e)
                    
    return Q, rwds


def deploy_model(Q, num_episodes):
    rwds = np.zeros(num_episodes)
    
    for episode in range(num_episodes):
        s = env.reset()
        total_rwd = 0
        
        while tip(s) > -1:
            best_a = torch.argmax(Q(torch.Tensor(s)))
            s, rwd, done, _ = env.step(best_a)
            env.render()
            total_rwd += rwd
        
        rwds[episode] = total_rwd
    
    return rwds

###############################################################################
# Run tests
###############################################################################
torch.manual_seed(0)
DS_Q = Net()
BFF_Q = copy.deepcopy(DS_Q)

ds_rwds  = np.zeros((rounds, max_episodes))
bff_rwds = np.zeros((rounds, max_episodes))

start = time.time()
for r in range(rounds):
    print(f'Running round {r}.')
    
    if method == 'bff':
        env.seed(r)
        np.random.seed(r)
        random.seed(r)
        BFF_Q, bff_rwds[r, :] = adam_BFF(max_episodes, 0.001, 50, Q = BFF_Q)
        torch.save(BFF_Q.state_dict(), f'acrobot_results/{experiment_n}/{method}_Q_{r}')
    
    if method == 'ds':
        env.seed(r)
        np.random.seed(r)
        random.seed(r)
        DS_Q, ds_rwds[r, :] = my_adam_DS(max_episodes, 0.001, 50, Q = DS_Q)
        env.close()
        torch.save(DS_Q.state_dict(), f'acrobot_results/{experiment_n}/{method}_Q_{r}')

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
    
    with open(f'acrobot_results/{experiment_n}/acrobot_ds_rwds.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in ds_rwds:
            writer.writerow(row)

if method == 'bff':
    ax.plot(x, bff_mean, label='bff')
    ax.fill_between(x, bff_mean - bff_sem, bff_mean + bff_sem, alpha = 0.3)
    
    with open(f'acrobot_results/{experiment_n}/acrobot_bff_rwds.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in bff_rwds:
            writer.writerow(row)

plt.title('Solve time per episode (acrobot)')
plt.xlabel('Episode')
plt.ylabel('Average reward +/- SEM')
plt.legend()
# plt.show()
plt.savefig(f'acrobot_results/{experiment_n}/{method}_acrobot_rwd.png')