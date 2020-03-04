"""
Test stochastic gradient approximation
"""
from trajectory import *
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
M = 100
g = 0.9

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
        if t % 10000 == 0:
            print(f't={t}')
        s = discrete_transition(S[t], A[t])[0]
        a = policy(s)
        S[t + 1] = s
        A[t + 1] = a
    return S, A


def full_grad(Q):
    return (g * bigP - np.eye(2 * N)) @ (np.diag(mu) @ (r + g * bigP.T @ Q - Q))

    
def discrete_uncorr_stoch_grad(S, A, Q, M=1):
    T = len(S) - 1
    batch = np.random.choice(range(T), M)
    G = np.zeros(Q.shape)
    
#    cur_freq = np.zeros(2 * N)
#    nxt_freq = np.zeros(2 * N)
#    new_freq = np.zeros(2 * N)
    for t in batch:
        cur_s = int(S[t])
        cur_a = int(A[t])
        nxt_s = int(discrete_transition(cur_s, cur_a)[0])
        nxt_a = int(policy(nxt_s))
        new_s = int(discrete_transition(cur_s, cur_a)[0])
        new_a = int(policy(new_s))
        
#        cur_freq[2 * cur_s + cur_a] += 1
#        nxt_freq[2 * nxt_s + nxt_a] += 1
#        new_freq[2 * new_s + new_a] += 1
            
        w1 = reward(cur_s) + g * Q[nxt_s, nxt_a] - Q[cur_s, cur_a]
        w2 = reward(cur_s) + g * Q[new_s, new_a] - Q[cur_s, cur_a]
            
        G[cur_s, cur_a] -= 0.5 * w1
        G[new_s, new_a] += 0.5 * g * w1
        G[cur_s, cur_a] -= 0.5 * w2
        G[nxt_s, nxt_a] += 0.5 * g * w2
    
    return G / M #, cur_freq / M, nxt_freq / M, new_freq / M


def uncorr_stoch_grad2(Q, M=1):
    G = np.zeros(Q.shape)
    
#    cur_freq = np.zeros(2 * N)
#    nxt_freq = np.zeros(2 * N)
#    new_freq = np.zeros(2 * N)
    for k in range(M):
        x = np.random.choice(range(2 * N), 1, p=mu)
        cur_s = int(x / 2)
        cur_a = int(x % 2)
        nxt_s = int(discrete_transition(cur_s, cur_a)[0])
        nxt_a = int(policy(nxt_s))
        new_s = int(discrete_transition(cur_s, cur_a)[0])
        new_a = int(policy(new_s))
    
#        cur_freq[2 * cur_s + cur_a] += 1
#        nxt_freq[2 * nxt_s + nxt_a] += 1
#        new_freq[2 * new_s + new_a] += 1
    
        w1 = reward(cur_s) + g * Q[nxt_s, nxt_a] - Q[cur_s, cur_a]
        w2 = reward(cur_s) + g * Q[new_s, new_a] - Q[cur_s, cur_a]
            
        G[cur_s, cur_a] -= 0.5 * w1
        G[new_s, new_a] += 0.5 * g * w1
        G[cur_s, cur_a] -= 0.5 * w2
        G[nxt_s, nxt_a] += 0.5 * g * w2
    
    return G / M #, cur_freq / M, nxt_freq / M, new_freq / M

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
#Q = np.zeros((N, 2))
#G = np.zeros((N, 2))
#M = 100000
#G = uncorr_stoch_grad2(Q, M)
#G = G.flatten()
#grad = full_grad(Q.flatten())
#print(np.linalg.norm(G))
#print(np.linalg.norm(grad))
#print(np.linalg.norm(G - grad))

## Stochastic gradient is unbiased when drawn from long enough fixed trajectory
#T = 5000000
#print('Generating trajectory...')
#S, A = simulate_discrete_trajectory(T)
#print('Finished generating trajectory.')
#M = 100000
#Q = np.zeros((N, 2))
#G = discrete_uncorr_stoch_grad(S, A, Q, M)
#G = G.flatten()
#grad = full_grad(Q.flatten())
#print(np.linalg.norm(G))
#print(np.linalg.norm(grad))
#print(np.linalg.norm(G.flatten() - grad))

Q = np.zeros((N, 2))
T = 5000000
M = 50
num_iter = 50000
#print('Generating trajectory...')
now = time.time()
S, A = simulate_discrete_trajectory(T)
#print('Finished generating trajectory.')
#print(f'Elapsed time: {time.time() - now}')
errors = np.zeros(num_iter)
start = time.time()
for k in range(num_iter):
    if k > 0 and k % 10 == 0:
        running = time.time() - start
        ETA = running * (num_iter - k) / k
        ETA /= 60
#        print(f'Running step {k}.')
#        print(f'Estimated time until completion: {ETA} min')
    G = uncorr_stoch_grad(S, A, Q, M)
    Q -= lr * G
    errors[k] = np.linalg.norm(Q.flatten() - trueQ)
    if errors[k] < 0.01:
        errors = errors[:k + 1]
        break
conv = len(errors)
print(f'Converged in {conv} steps.')
print(f'Final error: {errors[len(errors) - 1]}')
print(f'(Norm of true Q is {np.linalg.norm(trueQ)})')
plt.plot(range(len(errors)), errors)
plt.savefig('unbiased_sgd.png')
plt.figure()
plt.plot(range(64), trueQ)
plt.plot(range(64), Q.flatten())
plt.savefig('learned_Q.png')