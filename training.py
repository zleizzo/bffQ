"""
Training methods for BFFQ:
Uncorrelated, BFF, and double sampling
"""

from trajectory import *
import numpy as np
import random

g = 0.9

def uncorr_stoch_grad(S, A, Q, M=1):
    T = len(S) - 1
    batch = np.random.choice(range(T), M)
    G = np.zeros(Q.shape)
    
    for t in batch:
        cur_s = int(S[t])
        cur_a = int(A[t])
        nxt_s = int(S[t + 1])
        nxt_a = int(A[t + 1])
        new_s = transition(cur_s, cur_a)
        new_a = int(policy(new_s))
        
        w1 = reward(cur_s) + g * Q[nxt_s, nxt_a] - Q[cur_s, cur_a]
        w2 = reward(cur_s) + g * Q[new_s, new_a] - Q[cur_s, cur_a]
        
        G[cur_s, cur_a] -= 0.5 * w1
        G[new_s, new_a] += 0.5 * g * w1
        G[cur_s, cur_a] -= 0.5 * w2
        G[nxt_s, nxt_a] += 0.5 * g * w2
    
    return G / M

#def err(Q):