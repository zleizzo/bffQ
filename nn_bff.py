"""
Section 6.2 of BFFQ
"""
from scipy.linalg import null_space
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.cos(self.fc1(x))
        x = torch.cos(self.fc2(x))
        x = self.fc3(x)
        return x


sigma    = 0.2
epsilon  = 2 * np.pi / 32
sqrt_eps = np.sqrt(epsilon)
g        = 0.9


def transition(s, a):
    delta_s = a * epsilon + sigma * np.random.normal() * sqrt_eps
    return (s + delta_s) % (2 * np.pi)


def reward(s):
    return np.sin(s) + 1


def map_to_input(s):
    return torch.tensor([np.cos(s), np.sin(s)])


def policy_vec(s):
    return np.array([0.5 + np.sin(s) / 5, 0.5 - np.sin(s) / 5])


def policy(s):
    if np.random.rand() <= 0.5 + np.sin(s) / 5:
        return 0
    else:
        return 1


def simulate_trajectory(T = 1000000, s0 = 0):
    S = np.zeros(T + 1)
    A = np.zeros(T + 1)
    
    S[0] = s0
    A[0] = policy(s0)
    for t in range(T):
        if t % 100000 == 0:
            print(f'Simulating t = {t}')
        s = transition(S[t], int(A[t]))
        a = policy(s)
        S[t + 1] = s
        A[t + 1] = a
    return S, A


def compute_j(cur_s, cur_a, nxt_s, Q):
    pi    = policy_vec(nxt_s)
    rwd   = reward(nxt_s)
    cur_s = map_to_input(cur_s)
    nxt_s = map_to_input(nxt_s)
    return rwd + g * sum([pi[a] * Q(nxt_s)[a] for a in range(2)]) - Q(cur_s)[cur_a]


def compute_grad_j(cur_s, cur_a, nxt_s, Q):
    Q.zero_grad()
    computation = compute_j(cur_s, cur_a, nxt_s, Q)
    computation.backward()
    return [w.grad.data for w in Q.parameters()]


#def compute_grad_j(cur_s, cur_a, nxt_s, Q):
#    pi = policy_vec(nxt_s)
#    nxt_s = map_to_input(nxt_s)
#    cur_s = map_to_input(cur_s)
#    
#    Q.zero_grad()
#    for a in range(2):
#        computation = g * pi[a] * Q(nxt_s)[a]
#        computation.backward()
#    
#    computation = -Q(cur_s)[cur_a]
#    computation.backward()
#        
#    return [w.grad for w in Q.parameters()]


def UB(S, A, learning_rate, batch_size, epochs):
    T = len(S) - 1
#    errors  = np.zeros(epochs * int(T / batch_size))
    
    start = time.time()
    Q = Net()

    for epoch in range(epochs):
        print(f'Running epoch {epoch}.')
        if epoch > 0:
            print(f'ETA: {((time.time() - start) * (epochs - epoch) / epoch) / 60} min')
        t = 0
        epoch_start = time.time()
        for k in range(int(T / batch_size)):
            if k % 1000 == 0 and k > 0:
                print(f'ETA for this epoch: {round(((time.time() - epoch_start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
            # Compute stochastic gradient
            grads = [torch.zeros(w.shape) for w in Q.parameters()]
            for i in range(batch_size):
                cur_s = S[t]
                cur_a = int(A[t])
                nxt_s = S[t + 1]
                new_s = transition(cur_s, cur_a)
                
                j      = compute_j(cur_s, cur_a, nxt_s, Q)
                grad_j = compute_grad_j(cur_s, cur_a, new_s, Q)
                
                for l in range(len(grads)):
                    grads[l] += (j / batch_size) * grad_j[l]
                
            for w, grad in zip(Q.parameters(), grads):
                w.data.sub_(learning_rate * grad)
    
    return Q


def monte_carlo(s, a, tol = 0.001, reps = 1000):
    
    # T is defined so that the total reward incurred from time T to infinity is
    # at most tol.
    R_max = 2
    T = int(np.log((1 - g) * tol / R_max) / np.log(g)) + 1
    
    total = 0
    for r in range(reps):
        s_cur = s
        a_cur = a
        discount = 1
        for t in range(1, T):
            s_cur = transition(s_cur, a_cur)
            total += reward(s_cur) * discount
            a_cur = policy(s_cur)
            discount *= g
    empirical_avg = total / reps
    return empirical_avg


def MC(tol = 0.001, reps = 1000, divisions = 50):
    
    Q = np.zeros((divisions, 2))
    for i, s in zip(range(divisions), np.linspace(0, 2 * np.pi, divisions)):
        for a in range(2):
            print(f'Computing Q({s}, {a})...')
            Q[i, a] = monte_carlo(s, a, tol, reps)
    return Q


T             = 1000000
learning_rate = 0.1
batch_size    = 50
epochs        = 4


S, A = simulate_trajectory(T)
Q_UB = UB(S, A, learning_rate, batch_size, epochs)
Q_MC = MC()

x = np.linspace(0, 2 * np.pi)
z = torch.stack([map_to_input(s) for s in x])
ub = Q_UB(z).detach()
mc = Q_MC

plt.figure()
plt.subplot(1,2,1)
plt.plot(x, ub[:, 0], label='ub')
plt.plot(x, mc[:, 0], label='mc')
plt.title('Q, action 0')
plt.legend()

plt.subplot(1,2,2)
plt.plot(x, ub[:, 1], label='ub')
plt.plot(x, mc[:, 1], label='mc')
plt.title('Q, action 1')
plt.legend()

plt.savefig('bff_nn_plots.png')