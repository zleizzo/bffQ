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
dt         = 1

T          = 5000000
lr         = 0.1
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


def transition(s, a):
    drift     = actions[a]
    diffusion = np.random.randn() * sigma
    new_state = ( s * grid_size + (drift * dt) + (diffusion * np.sqrt(dt)) ) % (2 * np.pi) # keep s in [0, 2pi)
    return snap_to_grid(new_state)


def simulate_trajectory(T, s0=0):
    S = np.zeros(T + 1)
    A = np.zeros(T + 1)
    
    S[0] = s0
    A[0] = policy(s0)
    for t in range(T):
        if t % 100000 == 0:
            print(f'Simulating t = {t}')
        s = transition(S[t], int(A[t]))
        a = policy(s)
        S[t + 1] = s
        A[t + 1] = a
    return S, A


def uniform_SGD(Q_init = np.zeros((N, 2)), batch_size=50, T=5000000):
    Q = Q_init.copy()
    
    start = time.time()
    for k in range(int(T / batch_size)):
        if k % 1000 == 0 and k > 0:
            print(f'ETA: {round(((time.time() - start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
        G = np.zeros((N, 2))
        for i in range(batch_size):
            cur_s = np.random.randint(0, N)
            cur_a = np.random.randint(0, 2)
            nxt_s = transition(cur_s, cur_a)
            new_s = transition(cur_s, cur_a)
            
            pi_nxt = policy_vec(nxt_s)
            pi_new = policy_vec(new_s)
            
            w1 = reward(nxt_s) + g * np.dot(Q[nxt_s, :], pi_nxt) - Q[cur_s, cur_a]
            w2 = reward(new_s) + g * np.dot(Q[new_s, :], pi_new) - Q[cur_s, cur_a]
            
            G[cur_s, cur_a] -= 0.5 * w1
            G[cur_s, cur_a] -= 0.5 * w2
            for a in range(len(actions)):
                G[new_s, a] += 0.5 * pi_new[a] * g * w1
                G[nxt_s, a] += 0.5 * pi_nxt[a] * g * w2
                
        # Update Q
        Q -= (lr / batch_size) * G
    
    return Q


def unbiased_SGD_2(S, A, trueQ = None, Q_init = np.zeros((N, 2)), batch_size=1, epochs=1):
    T = len(S) - 1
    errors = np.zeros(epochs * int(T / batch_size))
    
    start = time.time()
    Q = Q_init.copy()

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
                new_s = int(transition(cur_s, cur_a))
                
                pi_nxt = policy_vec(nxt_s)
                pi_new = policy_vec(new_s)
                    
                w1 = reward(nxt_s) + g * np.dot(Q[nxt_s, :], pi_nxt) - Q[cur_s, cur_a]
                w2 = reward(new_s) + g * np.dot(Q[new_s, :], pi_new) - Q[cur_s, cur_a]
                    
                G[cur_s, cur_a] -= 0.5 * w1
                G[cur_s, cur_a] -= 0.5 * w2
                for a in range(len(actions)):
                    G[new_s, a] += 0.5 * pi_new[a] * g * w1
                    G[nxt_s, a] += 0.5 * pi_nxt[a] * g * w2
                    
                t += 1
            # Update Q
            Q -= (lr / batch_size) * G
            if trueQ is not None:
                errors[epoch * int(T / batch_size) + k] = np.linalg.norm(Q.flatten() - trueQ.flatten())
    
    return Q, errors


def BFFQ(S, A, trueQ, Q_init = np.zeros((N, 2)), batch_size = 1, epochs = 1):
    
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
                    
                w1 = reward(nxt_s) + g * np.dot(Q[nxt_s, :], pi_nxt) - Q[cur_s, cur_a]
                w2 = reward(new_s) + g * np.dot(Q[new_s, :], pi_new) - Q[cur_s, cur_a]
                    
                G[cur_s, cur_a] -= 0.5 * w1
                G[cur_s, cur_a] -= 0.5 * w2
                for a in range(len(actions)):
                    G[new_s, a] += 0.5 * pi_new[a] * g * w1
                    G[nxt_s, a] += 0.5 * pi_nxt[a] * g * w2
                    
                t += 1
            # Update Q        
            Q -= (lr / batch_size) * G
            errors[epoch * int(T / batch_size) + k] = np.linalg.norm(Q.flatten() - trueQ.flatten())
    
    return Q, errors


def double_sampling(S, A, trueQ, Q_init = np.zeros((N, 2)), batch_size = 1, epochs = 1):
    
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
                    
                w1 = reward(nxt_s) + g * np.dot(Q[nxt_s, :], pi_nxt) - Q[cur_s, cur_a]
                w2 = reward(new_s) + g * np.dot(Q[new_s, :], pi_new) - Q[cur_s, cur_a]
                    
                G[cur_s, cur_a] -= 0.5 * w1
                G[cur_s, cur_a] -= 0.5 * w2
                for a in range(len(actions)):
                    G[new_s, a] += 0.5 * pi_new[a] * g * w1
                    G[nxt_s, a] += 0.5 * pi_nxt[a] * g * w2
                    
                t += 1
            # Update Q        
            Q -= (lr / batch_size) * G
            errors[epoch * int(T / batch_size) + k] = np.linalg.norm(Q.flatten() - trueQ.flatten())
    
    return Q, errors


def monte_carlo(s, a, tol = 0.001, reps = 1000):
    
    # T is defined so that the total reward incurred from time T to infinity is
    # at most tol.
    R_max = 2
    T = int(np.log((1 - g) * tol / R_max) / np.log(g)) + 1
    
    total = 0
    for r in range(reps):
        s_cur = s
        a_cur = a
        discount = 1
        for t in range(1, T):
            s_cur = transition(s_cur, a_cur)
            total += reward(s_cur) * discount
            a_cur = policy(s_cur)
            discount *= g
    empirical_avg = total / reps
    return empirical_avg


def monte_carlo_Q(tol = 0.001, reps = 100000):
    
    Q = np.zeros((N, 2))
    for s in range(N):
        for a in range(2):
            print(f'Computing Q({s}, {a})...')
            Q[s, a] = monte_carlo(s, a, tol, reps)
    return Q


###############################################################################
# Run experiment
###############################################################################
S_long, A_long = simulate_trajectory(3 * T)
#Q_actual, _ = unbiased_SGD_2(S_long, A_long, batch_size = 150, epochs = 2)
Q_actual = uniform_SGD()

S, A = simulate_trajectory(T)
Q_UB, errors_UB   = unbiased_SGD_2(S, A, trueQ = Q_actual, batch_size = batch_size, epochs = epochs)
Q_BFF, errors_BFF = BFFQ(S, A, Q_actual, batch_size = batch_size, epochs = epochs)
Q_DS, errors_DS   = double_sampling(S, A, Q_actual, batch_size = batch_size, epochs = epochs)
Q_MC              = monte_carlo_Q()

initial_error  = np.linalg.norm(np.zeros(N * 2) - Q_actual.flatten())
rel_errors_UB  = [err / initial_error for err in errors_UB]
rel_errors_BFF = [err / initial_error for err in errors_BFF]
rel_errors_DS  = [err / initial_error for err in errors_DS]

log_errors_UB  = [np.log10(err) for err in rel_errors_UB]
log_errors_BFF = [np.log10(err) for err in rel_errors_BFF]
log_errors_DS  = [np.log10(err) for err in rel_errors_DS]

k = min([len(log_errors_UB), len(log_errors_BFF), len(log_errors_DS)])

plt.figure()
plt.plot(range(k), log_errors_UB[:k], label='ub')
plt.plot(range(k), log_errors_BFF[:k], label='bff')
plt.plot(range(k), log_errors_DS[:k], label='ds')
plt.xlabel('Iteration')
plt.ylabel('Relative error decay (log10 scale)')
plt.title('Relative training error decay')
plt.legend()
plt.savefig('errors.png')

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(range(N), Q_actual[:, 0], label='true')
plt.plot(range(N), Q_UB[:, 0], label='ub')
plt.plot(range(N), Q_BFF[:, 0], label='bff')
plt.plot(range(N), Q_DS[:, 0], label='ds')
plt.plot(range(N), Q_MC[:, 0], label='mc')
plt.xlabel('(state, action) pair')
plt.ylabel('Q value')
plt.title('Learned Q function, action 0')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(N), Q_actual[:, 1], label='true')
plt.plot(range(N), Q_UB[:, 1], label='ub')
plt.plot(range(N), Q_BFF[:, 1], label='bff')
plt.plot(range(N), Q_DS[:, 1], label='ds')
plt.plot(range(N), Q_MC[:, 1], label='mc')
plt.xlabel('(state, action) pair')
plt.ylabel('Q value')
plt.title('Learned Q function, action 1')
plt.legend()
plt.savefig('learned_q.png')