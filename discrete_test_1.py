from scipy.linalg import null_space
import numpy as np
import matplotlib.pyplot as plt
import time
import random
###############################################################################
# Define parameters
###############################################################################
g          = 0.9
N          = 32
sigma      = 1
actions    = [2 * np.pi / N, -2 * np.pi / N]
grid_size  = 2 * np.pi / N

T          = 5000000
lr         = 0.5
epochs     = 2
batch_size = 50

random.seed(0)
###############################################################################
# Define methods for generating trajectory
###############################################################################
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


###############################################################################
# Define Markov chain dynamics
###############################################################################
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


###############################################################################
# Define helper methods
###############################################################################
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
        if t % 100000 == 0:
            print(f'Simulating t = {t}')
        s = discrete_transition(S[t], A[t])[0]
        a = policy(s)
        S[t + 1] = s
        A[t + 1] = a
    return S, A


def unbiased_SGD(S, A, Q_init = np.zeros((N, 2)), batch_size=1, epochs=1):
    T = len(S) - 1
    errors = np.zeros(epochs * int(T / batch_size))
    start = time.time()
    Q = Q_init
    for epoch in range(epochs):
        print(f'Running epoch {epoch}.')
        if epoch > 0:
            print(f'ETA: {((time.time() - start) * (epochs - epoch) / epoch) / 60} min')
        t = 0
        epoch_start = time.time()
        for k in range(int(T / batch_size)):
            if k % 1000 == 0 and k > 0:
                print(f'ETA for this epoch: {round(((time.time() - epoch_start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
            # Compute stochastic gradient
            G = np.zeros((N, 2))
            for i in range(batch_size):
                cur_s = int(S[t])
                cur_a = int(A[t])
                nxt_s = int(S[t + 1])
                new_s = int(discrete_transition(cur_s, cur_a)[0])
                
                pi_nxt = policy_vec(nxt_s)
                pi_new = policy_vec(new_s)
                    
                w1 = reward(cur_s) + g * np.dot(Q[nxt_s, :], pi_nxt) - Q[cur_s, cur_a]
                w2 = reward(cur_s) + g * np.dot(Q[new_s, :], pi_new) - Q[cur_s, cur_a]
                    
                G[cur_s, cur_a] -= 0.5 * w1
                G[cur_s, cur_a] -= 0.5 * w2
                for a in range(len(actions)):
                    G[new_s, a] += 0.5 * pi_new[a] * g * w1
                    G[nxt_s, a] += 0.5 * pi_nxt[a] * g * w2
                    
                t += 1
            # Update Q        
            Q -= (lr / batch_size) * G
            errors[epoch * int(T / batch_size) + k] = np.linalg.norm(Q.flatten() - trueQ)
    
    return Q, errors


def BFFQ(S, A, Q_init = np.zeros((N, 2)), batch_size = 1, epochs = 1):
    
    T = len(S) - 3
    errors = np.zeros(epochs * int(T / batch_size))
    start = time.time()
    Q = Q_init
    for epoch in range(epochs):
        print(f'Running epoch {epoch}.')
        if epoch > 0:
            print(f'ETA: {((time.time() - start) * (epochs - epoch) / epoch) / 60} min')
        t = 0
        epoch_start = time.time()
        for k in range(int(T / batch_size)):
            if k % 1000 == 0 and k > 0:
                print(f'ETA for this epoch: {round(((time.time() - epoch_start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
            # Compute stochastic gradient
            G = np.zeros((N, 2))
            for i in range(batch_size):
                cur_s = int(S[t])
                cur_a = int(A[t])
                nxt_s = int(S[t + 1])
                new_s = int(S[t] + (S[t + 2] - S[t + 1])) % N
                
                pi_nxt = policy_vec(nxt_s)
                pi_new = policy_vec(new_s)
                    
                w1 = reward(cur_s) + g * np.dot(Q[nxt_s, :], pi_nxt) - Q[cur_s, cur_a]
                w2 = reward(cur_s) + g * np.dot(Q[new_s, :], pi_new) - Q[cur_s, cur_a]
                    
                G[cur_s, cur_a] -= 0.5 * w1
                G[cur_s, cur_a] -= 0.5 * w2
                for a in range(len(actions)):
                    G[new_s, a] += 0.5 * pi_new[a] * g * w1
                    G[nxt_s, a] += 0.5 * pi_nxt[a] * g * w2
                    
                t += 1
            # Update Q        
            Q -= (lr / batch_size) * G
            errors[epoch * int(T / batch_size) + k] = np.linalg.norm(Q.flatten() - trueQ)
    
    return Q, errors


def double_sampling(S, A, Q_init = np.zeros((N, 2)), batch_size = 1, epochs = 1):
    
    T = len(S)
    errors = np.zeros(epochs * int(T / batch_size))
    start = time.time()
    Q = Q_init
    for epoch in range(epochs):
        print(f'Running epoch {epoch}.')
        if epoch > 0:
            print(f'ETA: {((time.time() - start) * (epochs - epoch) / epoch) / 60} min')
        t = 0
        epoch_start = time.time()
        for k in range(int(T / batch_size)):
            if k % 1000 == 0 and k > 0:
                print(f'ETA for this epoch: {round(((time.time() - epoch_start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
            # Compute stochastic gradient
            G = np.zeros((N, 2))
            for i in range(batch_size):
                cur_s = int(S[t])
                cur_a = int(A[t])
                nxt_s = int(S[t + 1])
                new_s = int(S[t + 1])
                
                pi_nxt = policy_vec(nxt_s)
                pi_new = policy_vec(new_s)
                    
                w1 = reward(cur_s) + g * np.dot(Q[nxt_s, :], pi_nxt) - Q[cur_s, cur_a]
                w2 = reward(cur_s) + g * np.dot(Q[new_s, :], pi_new) - Q[cur_s, cur_a]
                    
                G[cur_s, cur_a] -= 0.5 * w1
                G[cur_s, cur_a] -= 0.5 * w2
                for a in range(len(actions)):
                    G[new_s, a] += 0.5 * pi_new[a] * g * w1
                    G[nxt_s, a] += 0.5 * pi_nxt[a] * g * w2
                    
                t += 1
            # Update Q        
            Q -= (lr / batch_size) * G
            errors[epoch * int(T / batch_size) + k] = np.linalg.norm(Q.flatten() - trueQ)
    
    return Q, errors



###############################################################################
# Run experiment
###############################################################################
S, A = simulate_discrete_trajectory(T)

Q_UB, errors_UB   = unbiased_SGD(S, A, batch_size = batch_size, epochs = epochs)
Q_BFF, errors_BFF = BFFQ(S, A, batch_size = batch_size, epochs = epochs)
Q_DS, errors_DS   = double_sampling(S, A, batch_size = batch_size, epochs = epochs)

initial_error  = np.linalg.norm(np.zeros(N * 2) - trueQ.flatten())
rel_errors_UB  = [err / initial_error for err in errors_UB]
rel_errors_BFF = [err / initial_error for err in errors_BFF]
rel_errors_DS  = [err / initial_error for err in errors_DS]

log_errors_UB  = [np.log10(err) for err in rel_errors_UB]
log_errors_BFF = [np.log10(err) for err in rel_errors_BFF]
log_errors_DS  = [np.log10(err) for err in rel_errors_DS]

k = min([len(log_errors_UB), len(log_errors_BFF), len(log_errors_DS)])

plt.plot(range(k), log_errors_UB[:k], label='ub')
plt.plot(range(k), log_errors_BFF[:k], label='bff')
plt.plot(range(k), log_errors_DS[:k], label='ds')
plt.xlabel('Iteration')
plt.ylabel('Relative error decay (log10 scale)')
plt.title('Relative training error decay')
plt.legend()

plt.figure()
plt.plot(range(2 * N), trueQ.flatten(), label='true')
plt.plot(range(2 * N), Q_UB.flatten(), label='ub')
plt.plot(range(2 * N), Q_BFF.flatten(), label='bff')
plt.plot(range(2 * N), Q_DS.flatten(), label='ds')
plt.xlabel('(state, action) pair')
plt.ylabel('Q value')
plt.title('Learned Q function')
plt.legend()