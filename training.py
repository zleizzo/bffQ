"""
Training methods for BFFQ:
Uncorrelated, BFF, and double sampling
"""

from trajectory import *
import numpy as np
import random

T = 10
g = 0.9 # gamma
N = 32
grid_size = 2 * np.pi / N

S, A = simulate_trajectory(T)

def uncorr_stoch_grad(S, A, Q):
    T = len(S)
    t = random.sample(range(T), 1)[0]
    
    pi1 = policy_vec(S[(t + 1) % T])
    
    G = np.zeros(Q.shape)
    G[int(S[t]), int(A[t])] += reward(S[t]) + g * sum([pi1[b] * Q[int(S[(t + 1) % T]), b] for b in range(2)]) - Q[int(S[t]), int(A[t])]
    
    s, _ = transition(S[t])
    pi2  = policy_vec(s)
    for  a in range(2):
        G[s, a] += pi2[a] * g * (reward(S[t]) + g * sum([pi1[b] * Q[int(S[(t + 1) % T]), b] for b in range(2)]) - Q[int(S[t]), int(A[t])])
    
    return G