"""
Test each retraining method.
"""

from trajectory import *
from training import *
import numpy as np

T = 10
g = 0.9 # gamma
N = 32
grid_size = 2 * np.pi / N
num_iter = 100
lr = 0.1

S, A = simulate_trajectory(T)
Q = np.zeros((N, 2))

for k in range(num_iter):
    G = uncorr_stoch_grad(S, A, Q)
    Q -= lr * G