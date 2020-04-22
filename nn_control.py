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

def e_greedy(choices, e):
    """
    Returns the index for an e-greedy choice from choices
    """
    if np.random.rand() <= e:
        while True:
            i = np.random.randint(0, len(choices))
            if i != torch.argmax(choices):
                return i
    else:
        return torch.argmax(choices)


def L2_error(Q1graph, Q2graph):
    """
    Computes the L2 norm of Q1 - Q2 with n subdivisions of [0, 2pi).
    Q2graph should be given as tensors to save computation time.
    (Q2 will remain fixed during training.)
    """
    n = len(Q1graph)
    with torch.no_grad():
        norm = torch.norm(Q1graph[:n - 1] - Q2graph[:n - 1])
        norm *= np.sqrt(2 * np.pi / (n - 1))
    return norm.numpy()


def compute_j(cur_s, cur_a, nxt_s, Q, e = 0.1):
    rwd   = reward(nxt_s)
    cur_s = map_to_input(cur_s)
    nxt_s = map_to_input(nxt_s)
    
    nxt_a = e_greedy(Q(nxt_s), e)
    return rwd + g * Q(nxt_s)[nxt_a] - Q(cur_s)[cur_a]


def compute_grad_j(cur_s, cur_a, nxt_s, Q, e = 0.1):
    Q.zero_grad()
    computation = compute_j(cur_s, cur_a, nxt_s, Q)
    computation.backward()
    return [w.grad.data for w in Q.parameters()]


def unif_UB(T, learning_rate, batch_size, e = 0.1, Q = Net(), trueQgraph = None):
    
    start = time.time()
    errors = np.zeros(int(T / batch_size))
    x = np.linspace(0, 2 * np.pi)
    z = torch.stack([map_to_input(s) for s in x])
    
    print('Starting uniform UB SGD...')
    for k in range(int(T / batch_size)):
        if k % 100 == 0 and k > 0:
            print(f'ETA: {round(((time.time() - start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
        grads = [torch.zeros(w.shape) for w in Q.parameters()]
        for i in range(batch_size):
            cur_s = np.random.rand() * 2 * np.pi
            cur_a = np.random.randint(0, 2)
            nxt_s = transition(cur_s, cur_a)
            new_s = transition(cur_s, cur_a)
            
            j      = compute_j(cur_s, cur_a, nxt_s, Q, e)
            grad_j = compute_grad_j(cur_s, cur_a, new_s, Q, e)
                
            for l in range(len(grads)):
                grads[l] += (j / batch_size) * grad_j[l]
                
        for w, grad in zip(Q.parameters(), grads):
            w.data.sub_(learning_rate * grad)
            
        if trueQgraph is not None:
            with torch.no_grad():
                Qgraph = Q(z)
                errors[k] = L2_error(Qgraph, trueQgraph)
    
    return Q, errors


def unif_DS(T, learning_rate, batch_size, e = 0.1, Q = Net(), trueQgraph = None):
    
    start = time.time()
    errors = np.zeros(int(T / batch_size))
    x = np.linspace(0, 2 * np.pi)
    z = torch.stack([map_to_input(s) for s in x])
    
    print('Starting uniform DS SGD...')
    for k in range(int(T / batch_size)):
        if k % 100 == 0 and k > 0:
            print(f'ETA: {round(((time.time() - start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
        grads = [torch.zeros(w.shape) for w in Q.parameters()]
        for i in range(batch_size):
            cur_s = np.random.rand() * 2 * np.pi
            cur_a = np.random.randint(0, 2)
            nxt_s = transition(cur_s, cur_a)
            new_s = nxt_s
            
            j      = compute_j(cur_s, cur_a, nxt_s, Q, e)
            grad_j = compute_grad_j(cur_s, cur_a, new_s, Q, e)
                
            for l in range(len(grads)):
                grads[l] += (j / batch_size) * grad_j[l]
                
        for w, grad in zip(Q.parameters(), grads):
            w.data.sub_(learning_rate * grad)
        
        if trueQgraph is not None:
            with torch.no_grad():
                Qgraph = Q(z)
                errors[k] = L2_error(Qgraph, trueQgraph)
    
    return Q, errors


def unif_BFF(T, learning_rate, batch_size, e = 0.1, Q = Net(), trueQgraph = None):
    
    start = time.time()
    errors = np.zeros(int(T / batch_size))
    x = np.linspace(0, 2 * np.pi)
    z = torch.stack([map_to_input(s) for s in x])
    
    print('Starting uniform UB SGD...')
    for k in range(int(T / batch_size)):
        if k % 100 == 0 and k > 0:
            print(f'ETA: {round(((time.time() - start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
        grads = [torch.zeros(w.shape) for w in Q.parameters()]
        for i in range(batch_size):
            cur_s = np.random.rand() * 2 * np.pi
            cur_a = np.random.randint(0, 2)
            
            nxt_s = transition(cur_s, cur_a)
            nxt_a = policy(nxt_s)
            
            new_s = cur_s + (transition(nxt_s, nxt_a) - nxt_s)            
            
            j      = compute_j(cur_s, cur_a, nxt_s, Q, e)
            grad_j = compute_grad_j(cur_s, cur_a, new_s, Q, e)
                
            for l in range(len(grads)):
                grads[l] += (j / batch_size) * grad_j[l]
                
        for w, grad in zip(Q.parameters(), grads):
            w.data.sub_(learning_rate * grad)
            
        if trueQgraph is not None:
            with torch.no_grad():
                Qgraph = Q(z)
                errors[k] = L2_error(Qgraph, trueQgraph)
                
    return Q, errors


def monte_carlo(s, a, trueQ, e = 0.1, tol = 0.001, reps = 100):
    
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
            a_cur = e_greedy(trueQ(map_to_input(s_cur)), e)
            discount *= g
    empirical_avg = total / reps
    return empirical_avg


def MC(trueQ, e = 0.1, tol = 0.001, reps = 1000, divisions = 50):
    
    Q = np.zeros((divisions, 2))
    for i, s in zip(range(divisions), np.linspace(0, 2 * np.pi, divisions)):
        for a in range(2):
            print(f'Computing Q({s}, {a})...')
            Q[i, a] = monte_carlo(s, a, trueQ, e, tol, reps)
    return Q


random.seed(0)

T             = 1000000
learning_rate = 0.1
batch_size    = 50
epochs        = 4
e             = 0.1

trueQ, _      = unif_UB(5 * T, learning_rate, batch_size, e, Net())

x = np.linspace(0, 2 * np.pi)
z = torch.stack([map_to_input(s) for s in x])
trueQgraph = trueQ(z).detach()

Q_UB,  e_UB  = unif_UB(T, learning_rate, batch_size, e, Net(), trueQgraph)
Q_DS,  e_DS  = unif_DS(T, learning_rate, batch_size, e, Net(), trueQgraph)
Q_BFF, e_BFF = unif_BFF(T, learning_rate, batch_size, e, Net(), trueQgraph)
Q_MC         = MC(trueQ, e)

mc  = Q_MC
ub  = Q_UB(z).detach()
ds  = Q_DS(z).detach()
bff = Q_BFF(z).detach()
true = trueQ(z).detach()

plt.figure()
plt.subplot(1,2,1)
plt.plot(x, true[:, 0], label='true', color='b')
plt.plot(x, mc[:, 0], label='mc', color='c')
plt.plot(x, ub[:, 0], label='ub', color='m')
plt.plot(x, ds[:, 0], label='ds', color='r')
plt.plot(x, bff[:, 0], label='bff', color='g')
plt.title('Q, action 0')
plt.legend()

plt.subplot(1,2,2)
plt.plot(x, true[:, 1], label='true', color='b')
plt.plot(x, mc[:, 1], label='mc', color='c')
plt.plot(x, ub[:, 1], label='ub', color='m')
plt.plot(x, ds[:, 1], label='ds', color='r')
plt.plot(x, bff[:, 1], label='bff', color='g')
plt.title('Q, action 1')
plt.legend()
plt.savefig('control_q.png')



rel_e_UB  = [err / e_UB[0]  for err in e_UB]
rel_e_DS  = [err / e_DS[0]  for err in e_DS]
rel_e_BFF = [err / e_BFF[0] for err in e_BFF]

log_e_UB  = [np.log10(err) for err in rel_e_UB]
log_e_DS  = [np.log10(err) for err in rel_e_DS]
log_e_BFF = [np.log10(err) for err in rel_e_BFF]

plt.figure()
plt.plot(log_e_UB,  label='ub',  color='b')
plt.plot(log_e_DS,  label='ds',  color='r')
plt.plot(log_e_BFF, label='bff', color='g')
plt.xlabel('Iteration')
plt.ylabel('Relative error decay (log10 scale)')
plt.title('Relative training error decay, uniform (s, a) sampling')
plt.legend()
plt.savefig('control_error.png')