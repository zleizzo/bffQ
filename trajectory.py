"""
Simulate BFF trajectory

s always denotes the state (angle theta)
"""

import numpy as np

N      = 32
dt     = 1
sigma  = 0.5

grid_size = 2 * np.pi / N

a = [2 * np.pi / N, -2 * np.pi / N]

def reward(s):
    s *= grid_size
    return 1 + np.sin(s)


def policy(s):
    s *= grid_size
    if np.random.rand() <= 0.5 + np.sin(s) / 5:
        return 0
    else:
        return 1


def policy_vec(s):
    s *= grid_size
    return [0.5 + np.sin(s) / 5, 0.5 - np.sin(s) / 5]


def snap_to_grid(s):
    """
    Returns k s.t. s is nearest to 2pi * k / N
    Actual angle is snap_to_grid * grid_size
    """
    resid = s % grid_size
    if resid >= grid_size / 2:
        return int(round((s - resid) / grid_size, 0) + 1) % N
    else:
        return int(round((s - resid) / grid_size, 0)) % N


def transition(s):
    action = policy(s)
    drift  = a[action]
    diffusion = np.random.randn() * sigma
    new_state = ( s * grid_size + (drift * dt) + (diffusion * np.sqrt(dt)) ) % (2 * np.pi) # keep s in [0, 2pi)
    return snap_to_grid(new_state), action


def simulate_trajectory(T, s0=0):
    """
    Simulate the trajectory for T steps starting from s0.
    Returns two length T + 1 vectors S = (s0, s1, ..., sT), A = (a0, a1, ..., aT).
    """
    S = np.zeros(T + 1)
    A = np.zeros(T + 1)
    
    S[0] = s0
    A[0] = int(policy(s0))
    for t in range(T):
        s, a = transition(S[t] * grid_size)
        S[t + 1] = s
        A[t + 1] = a
    return S, A