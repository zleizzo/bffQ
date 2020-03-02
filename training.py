"""
Training methods for BFFQ:
Uncorrelated, BFF, and double sampling
"""

from trajectory import *
import numpy as np
import random

g = 0.9 # gamma
N = 32
grid_size = 2 * np.pi / N
M = 1

def uncorr_stoch_grad(S, A, Q, M=1):
    T = len(S) - 1
    batch = random.sample(range(T), M)
    G = np.zeros(Q.shape)
    for t in batch:
        G[int(S[t]), int(A[t])] -= reward(S[t]) + g * Q[int(S[t + 1]), int(A[t + 1])] - Q[int(S[t]), int(A[t])]
        s, a = transition(S[t])
        G[s, a] += g * (reward(S[t]) + g * Q[int(S[t + 1]), int(A[t + 1])] - Q[int(S[t]), int(A[t])])
    
    return G / M

#def err(Q):