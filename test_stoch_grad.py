"""
Test stochastic gradient approximation
"""
from trajectory import *
from training import *
from scipy.linalg import null_space
import numpy as np
import matplotlib.pyplot as plt
import time

N      = 32
sigma  = 1
grid_size = 2 * np.pi / N
actions = [2 * np.pi / N, -2 * np.pi / N]
num_iter = 100000
lr = 0.5
T = 1000000
M = 1000

# For testing purposes, define dynamics so that we can explicitly calulate P
dist = [(N / 2) - abs(k - N / 2) for k in range(N)]
p = np.array([2 ** (-d / sigma) for d in dist])
p /= sum(p)

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


v = null_space(bigP - np.eye(2 * N))
mu = v / sum(v)
mu = mu.flatten()
trueQ = np.linalg.solve(np.eye(N * 2) - g * bigP.T, r)

def discrete_transition(s, a):
    drift  = 1 - 2 * a
    diffusion = np.random.choice(range(N), 1, p=p)
    return (s + drift + diffusion) % N


def simulate_discrete_trajectory(T, s0=0):
    S = np.zeros(T + 1)
    A = np.zeros(T + 1)
    
    S[0] = s0
    A[0] = policy(s0)
    for t in range(T):
        s = discrete_transition(S[t], A[t])[0]
        a = policy(s)
        S[t + 1] = s
        A[t + 1] = a
    return S, A

def full_grad(Q):
    return (g * bigP - np.eye(2 * N)) @ (np.diag(mu) @ (r + g * bigP.T @ Q - Q))

# BEGIN TESTS

## Empirical trajectory matches analytical results
#T = 100000
#S, A = simulate_discrete_trajectory(T)
#empirical_dist = np.zeros((N, 2))
#for t in range(T):
#    empirical_dist[int(S[t]), int(A[t])] += 1
#
#empirical_dist /= T
#plt.plot(range(64), mu)
#plt.plot(range(64), empirical_dist.flatten())
    
## Full GD converges for Q function
#Q = np.zeros(2 * N)
#errors = np.zeros(num_iter)
#for k in range(num_iter):
#    G = full_grad(Q)
#    Q -= lr * G
#    errors[k] = np.linalg.norm(Q - trueQ)
#    if errors[k] < 0.01:
#        errors = errors[:k + 1]
#        break
#plt.plot(range(len(errors)), errors)
#plt.figure()
#plt.plot(range(64), Q)
#plt.plot(range(64), trueQ)

## Stochastic gradient is unbiased when not drawn from fixed trajectory
def uncorr_stoch_grad2(Q):
    G = np.zeros(Q.shape)
    x = np.random.choice(range(2 * N), 1, p=mu)
    cur_s = int(x / 2)
    cur_a = int(x % 2)
    nxt_s = discrete_transition(cur_s, cur_a)[0]
    nxt_a = int(policy(nxt_s))
    new_s = discrete_transition(cur_s, cur_a)[0]
    new_a = int(policy(new_s))
        
    w1 = reward(cur_s) + g * Q[nxt_s, nxt_a] - Q[cur_s, cur_a]
    w2 = reward(cur_s) + g * Q[new_s, new_a] - Q[cur_s, cur_a]
        
    G[cur_s, cur_a] -= 0.5 * w1
    G[new_s, new_a] += 0.5 * g * w1
    G[cur_s, cur_a] -= 0.5 * w2
    G[nxt_s, nxt_a] += 0.5 * g * w2
    
    return G

Q = np.zeros((N, 2))
G = np.zeros((N, 2))
M = 100000
for k in range(M):
    G += uncorr_stoch_grad2(Q)
G /= M
G = G.flatten()
grad = full_grad(Q.flatten())
print(np.linalg.norm(G))
print(np.linalg.norm(grad))
print(np.linalg.norm(G - grad))

T = 1000000
print('Generating trajectory...')
S, A = simulate_discrete_trajectory(T)
print('Done generating trajectory.')
freqs = np.zeros(2 * N)
for t in range(len(S)):
    freqs[int(2 * S[t] + A[t])] += 1
freqs /= len(S)
plt.plot(range(2 * N), mu)
plt.plot(range(2 * N), freqs)

G = uncorr_stoch_grad(S, A, Q, M)
true_grad = full_grad(Q.flatten())
print(np.linalg.norm(G.flatten()))
print(np.linalg.norm(true_grad))
print(np.linalg.norm(G.flatten() - true_grad))

#Q = np.zeros((N, 2))
#errors = np.zeros(num_iter)
#start = time.time()
#for k in range(num_iter):
#    if k > 0 and k % 100 == 0:
#        running = time.time() - start
#        ETA = running * (num_iter - k) / k
#        ETA /= 60
#        print(f'Running step {k}.')
#        print(f'Estimated time until completion: {ETA} min')
#    G = uncorr_stoch_grad(S, A, Q, M)
#    Q -= lr * G
#    errors[k] = np.linalg.norm(Q.flatten() - trueQ)
#    if errors[k] < 0.01:
#        errors = errors[:k + 1]
#        break
#conv = len(errors)
#print(f'Converged in {conv} steps.')
#plt.plot(range(len(errors)), errors)