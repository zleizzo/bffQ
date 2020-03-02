"""
Test stochastic gradient approximation
"""
from trajectory import *
from training import *
from scipy.linalg import null_space
import numpy as np
import matplotlib.pyplot as plt

N      = 32
sigma  = 2
grid_size = 2 * np.pi / N
actions = [2 * np.pi / N, -2 * np.pi / N]
num_iter = 100000
lr = 1

# For testing purposes, define dynamics so that we can explicitly calulate P
dist = [(N / 2) - abs(k - N / 2) for k in range(N)]
p = [np.exp(-d / sigma) for d in dist]
p /= sum(p)
print(sum(p))

P = np.zeros((N, N, 2))
for s in range(N):
    for t in range(N):
        P[(s + 1 + t) % N, s, 0] = p[t]
        P[(s - 1 + t) % N, s, 1] = p[t]

bigP = np.zeros((N * 2, N * 2))
for s in range(N):
    for a in range(2):
        for t in range(N):
            for b in range(2):
                pi = policy_vec(t)
                bigP[2 * t + b, 2 * s + a] = P[t, s, a] * pi[b]

r = np.zeros(N * 2)
for s in range(N):
    for a in range(2):
        r[2 * s + a] = reward(s)

trueQ = np.linalg.solve(np.eye(N * 2) - g * bigP, r)
v = null_space(bigP - np.eye(2 * N))
mu = v / sum(v)


def discrete_transition(s):
    action = policy(s)
    drift  = 1 - 2 * action
    diffusion = np.random.choice(range(N), 1, p=p)
    return (s + drift + diffusion) % N, action


def simulate_discrete_trajectory(T, s0=0):
    S = np.zeros(T + 1)
    A = np.zeros(T + 1)
    
    S[0] = s0
    A[0] = int(policy(s0))
    for t in range(T):
        s, a = discrete_transition(S[t])
        S[t + 1] = s
        A[t + 1] = a
    return S, A

def full_grad(Q):
    return (g * bigP - np.eye(2 * N)).T @ (np.diag(mu[:, 0]) @ (r + g * bigP @ Q - Q))

# BEGIN TESTS
    
S, A = simulate_discrete_trajectory(T)
#print(len(S))
Q = np.zeros(2 * N)
errors = np.zeros(num_iter)
for k in range(num_iter):
    G = full_grad(Q)
    Q -= lr * G
    errors[k] = np.linalg.norm(Q - trueQ)
    if errors[k] < 0.01:
        errors = errors[:k + 1]
        break
    
#Q = np.zeros((N, 2))
#errors = np.zeros(num_iter)
#for k in range(num_iter):
#    G = uncorr_stoch_grad(S, A, Q)
#    Q -= lr * G
#    errors[k] = np.linalg.norm(Q.flatten() - trueQ)
#    if errors[k] < 0.01:
#        errors = errors[:k + 1]
#        break

conv = len(errors)
print(f'Converged in {conv} steps.')
plt.plot(range(len(errors)), errors)