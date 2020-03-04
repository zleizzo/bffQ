"""
Testing unbiased gradient for value function
"""

from scipy.linalg import null_space
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(0)

N      = 32
sigma  = 2
grid_size = 2 * np.pi / N
actions = [2 * np.pi / N, -2 * np.pi / N]
num_iter = 100000
lr = 0.5
T = 1000000
M = 50
g = 0.9

grid_size = 2 * np.pi / N


dist = [(N / 2) - abs(k - N / 2) for k in range(N)]
p = [np.exp(-d / sigma) for d in dist]
p /= sum(p)

def policy_vec(s):
    s *= grid_size
    return [0.5 + np.sin(s) / 5, 0.5 - np.sin(s) / 5]

P0 = np.zeros((N, N))
P1 = np.zeros((N, N))
for s in range(N):
    for t in range(N):
        P0[(s + 1 + t) % N, s] = p[t]
        P1[(s - 1 + t) % N, s] = p[t]

P = np.zeros((N, N))
for s in range(N):
    pi = policy_vec(s)
    P[:, s] = pi[0] * P0[:, s] + pi[1] * P1[:, s]

def reward(s):
    s *= grid_size
    return 1 + np.sin(s)

r = np.array([reward(s) for s in range(N)])
mu = null_space(P - np.eye(N)).flatten()
mu /= sum(mu)

trueV = np.linalg.solve(np.eye(N) - g * P.T, r)

def transition(s):
    new = np.random.choice(range(N), 1, p=P[:, s].flatten())
    return new[0]

def gen_trajectory(T):
    S = np.zeros(T + 1)
    S[0] = 0
    for t in range(T):
        S[t + 1] = transition(int(S[t]))
    return S

def uncorr_stoch_grad(S, V, M):
    G = np.zeros(V.shape)
    batch = random.sample(range(T), M)
    
    for t in batch:
        cur = int(S[t])
        nxt = int(S[t + 1])
        new = int(transition(cur))
        
        w1 = reward(cur) + g * V[nxt] - V[cur]
        w2 = reward(cur) + g * V[new] - V[cur]
        
        G[cur] -= 0.5 * w1
        G[new] += 0.5 * g * w1
        G[cur] -= 0.5 * w2
        G[nxt] += 0.5 * g * w2
        
    return G / M

def uncorr_stoch_grad2(V):
    cur = np.random.choice(range(N), 1, p=mu)[0]
    nxt = transition(cur)
    new = transition(cur)
    
    w1 = reward(cur) + g * V[nxt] - V[cur]
    w2 = reward(cur) + g * V[new] - V[cur]
    
    G = np.zeros(N)
    
    G[cur] -= 0.5 * w1
    G[new] += 0.5 * g * w1
    G[cur] -= 0.5 * w2
    G[nxt] += 0.5 * g * w2
    
    return G

def grad(V):
    return (g * P - np.eye(N)) @ (np.diag(mu) @ (r + g * P.T @ V - V))


# BEGIN TESTS
#unbiased = np.zeros(N)
#V = np.zeros(N)
#for k in range(100000):
#    unbiased += uncorr_stoch_grad2(V)
#unbiased /= 100000
#true_grad = grad(V)
#print(np.linalg.norm(unbiased))
#print(np.linalg.norm(true_grad))
#print(np.linalg.norm(unbiased-true_grad))

#s = 0
#gap = np.zeros(100000)
#for i in range(100000):
#    new = transition(s)
#    gap[i] = (new - s) % N
#    s = new
#plt.plot(range(N), p)
#plt.hist(gap, bins=32, density=True)

print('Generating trajectory...')
S = gen_trajectory(T)
print('Done generating trajectory.')
plt.figure()
plt.plot(range(N), mu)
plt.hist(S, bins=N, density=True)

#V = np.zeros(N)
#G = uncorr_stoch_grad(S, V, 1000)
#full_grad = grad(V)
#print(np.linalg.norm(G))
#print(np.linalg.norm(full_grad))
#print(np.linalg.norm(G - full_grad))

V = np.zeros(N)
errors = np.zeros(num_iter)
for k in range(num_iter):
    G = grad(V)
    V -= lr * G
    errors[k] = np.linalg.norm(V - trueV)
    if errors[k] < 0.01:
        errors = errors[:k + 1]
        break
plt.figure()
plt.plot(range(len(errors)), errors)

V = np.zeros(N)
errors = np.zeros(num_iter)
for k in range(num_iter):
    G = G = uncorr_stoch_grad(S, V, M)
    V -= lr * G
    errors[k] = np.linalg.norm(V - trueV)
    if errors[k] < 0.01:
        errors = errors[:k + 1]
        break
plt.plot(range(len(errors)), errors)

plt.plot(range(N), V)
plt.plot(range(N), trueV)