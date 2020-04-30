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

T          = 10000000
lr         = 0.5
batch_size = 50

reps       = 1000

random.seed(0)
###############################################################################
# Define methods for generating trajectory
###############################################################################
def reward(s):
    """
    Compute the reward for state s.
    s is given as an integer in {0, 1, ..., N-1}.
    """
    s *= grid_size # Convert s to an angle.
    return 1 + np.sin(s)


def policy(s):
    """
    Chooses an action at state s based on the fixed policy pi.
    """
    s *= grid_size # Convert s to an angle.
    if np.random.rand() <= 0.5 + np.sin(s) / 5:
        return 0
    else:
        return 1
    

def policy_vec(s):
    """
    Computes the vector of action probabilities at state s for the fixed policy pi.
    """
    s *= grid_size # Convert s to an angle.
    return [0.5 + np.sin(s) / 5, 0.5 - np.sin(s) / 5]


def snap_to_grid(s):
    """
    Returns k such that s is nearest to k * 2pi / N.
    """
    resid = s % grid_size # Computes s mod 2pi / N
    if resid >= grid_size / 2:
        return int(round((s - resid) / grid_size, 0) + 1) % N # Mod N ensures state in {0, 1, ..., N-1}.
    else:
        return int(round((s - resid) / grid_size, 0)) % N # Mod N ensures state in {0, 1, ..., N-1}.


def transition(s, a):
    """
    Perform a state transition in the Markov chain from state s with action a.
    """
    drift     = actions[a] # Maps a = 0 to a drift of -2pi/N, a = 1 to a drift of +2pi/N.
    diffusion = np.random.randn() * sigma 
    
    new_state = ( s * grid_size + (drift * dt) + (diffusion * np.sqrt(dt)) ) % (2 * np.pi) # Keep s in [0, 2pi)
    return snap_to_grid(new_state) # Convert from angle back to one of the discrete states {0, 1, ..., N-1}.


# It seems like some states are never being visited. Check this?
def UB(T, trueQ = None, Q_init = np.zeros((N, 2)), batch_size=1, P = None, r = None):
    errors  = np.zeros(int(T / batch_size))
    bellman = np.zeros(int(T / batch_size))
    
    start = time.time()
    Q = Q_init.copy()

    print('Starting UB SGD...')
    
    # k denotes the k-th step of SGD.
    for k in range(int(T / batch_size)):
        # Runtime estimate.
        if k % 100 == 0 and k > 0:
            print(f'ETA: {round(((time.time() - start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
        
        # Compute the gradient on the next batch.
        # Relationship to notation in the paper:
        #   cur_s = s_m
        #   cur_a = a_m
        #   nxt_s = s_{m+1}
        #   new_s = s'_{m+1}
        G = np.zeros((N, 2))
        for i in range(batch_size):
            # Move to next point in the trajectory and select an e-greedy action.
            cur_s = np.random.randint(0, N)
            cur_a = np.random.randint(0, 2)
            
            # Generate the next step in the trajectory based on our current state and action.
            nxt_s = transition(cur_s, cur_a)
            
            # For UB SGD, generate an independent copy of the next state.
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
        
        Q -= (lr / batch_size) * G
        if trueQ is not None:
            errors[k] = np.linalg.norm(Q.flatten() - trueQ.flatten())
        if P is not None:
            bellman[k] = np.linalg.norm(r + (g * P.T - np.eye(2 * N)) @ Q.flatten())

    return Q, errors, bellman


def DS(T, trueQ = None, Q_init = np.zeros((N, 2)), batch_size=1, P = None, r = None):
    errors  = np.zeros(int(T / batch_size))
    bellman = np.zeros(int(T / batch_size))
    
    start = time.time()
    Q = Q_init.copy()

    print('Starting UB SGD...')
    
    
    # k denotes the k-th step of SGD.
    for k in range(int(T / batch_size)):
        # Runtime estimate.
        if k % 100 == 0 and k > 0:
            print(f'ETA: {round(((time.time() - start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
        
        # Compute the gradient on the next batch.
        # Relationship to notation in the paper:
        #   cur_s = s_m
        #   cur_a = a_m
        #   nxt_s = s_{m+1}
        #   new_s = s'_{m+1}
        G = np.zeros((N, 2))
        for i in range(batch_size):
            # Move to next point in the trajectory and select an e-greedy action.
            cur_s = np.random.randint(0, N)
            cur_a = np.random.randint(0, 2)
            
            # Generate the next step in the trajectory based on our current state and action.
            nxt_s = transition(cur_s, cur_a)
            
            # For UB SGD, generate an independent copy of the next state.
            new_s = nxt_s
            
            pi_nxt = policy_vec(nxt_s)
            pi_new = policy_vec(new_s)
            
            w1 = reward(nxt_s) + g * np.dot(Q[nxt_s, :], pi_nxt) - Q[cur_s, cur_a]
            w2 = reward(new_s) + g * np.dot(Q[new_s, :], pi_new) - Q[cur_s, cur_a]
            
            G[cur_s, cur_a] -= 0.5 * w1
            G[cur_s, cur_a] -= 0.5 * w2
            for a in range(len(actions)):
                G[new_s, a] += 0.5 * pi_new[a] * g * w1
                G[nxt_s, a] += 0.5 * pi_nxt[a] * g * w2
        
        Q -= (lr / batch_size) * G
        if trueQ is not None:
            errors[k] = np.linalg.norm(Q.flatten() - trueQ.flatten())
        if P is not None:
            bellman[k] = np.linalg.norm(r + (g * P.T - np.eye(2 * N)) @ Q.flatten())

    return Q, errors, bellman


def BFF(T, trueQ = None, Q_init = np.zeros((N, 2)), batch_size=1, P = None, r = None):
    errors  = np.zeros(int(T / batch_size))
    bellman = np.zeros(int(T / batch_size))
    
    start = time.time()
    Q = Q_init.copy()

    print('Starting UB SGD...')
    
    # Choose a random initial state
    cur_s = np.random.rand() * 2 * np.pi
    cur_a = policy(cur_s)
    
    nxt_s = transition(cur_s, cur_a)
    nxt_a = policy(nxt_s)
    
    ftr_s = transition(nxt_s, nxt_a)
    
    # k denotes the k-th step of SGD.
    for k in range(int(T / batch_size)):
        # Runtime estimate.
        if k % 100 == 0 and k > 0:
            print(f'ETA: {round(((time.time() - start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
        
        # Compute the gradient on the next batch.
        # Relationship to notation in the paper:
        #   cur_s = s_m
        #   cur_a = a_m
        #   nxt_s = s_{m+1}
        #   new_s = s'_{m+1}
        G = np.zeros((N, 2))
        for i in range(batch_size):
            # Move to next point in the trajectory and select an e-greedy action.
            cur_s = np.random.randint(0, N)
            cur_a = np.random.randint(0, 2)
            
            nxt_s = transition(cur_s, cur_a)
            nxt_a = policy(nxt_s)
            
            ftr_s = transition(nxt_s, nxt_a)
            
            new_s = (cur_s + (ftr_s - nxt_s)) % N
            
            pi_nxt = policy_vec(nxt_s)
            pi_new = policy_vec(new_s)
            
            w1 = reward(nxt_s) + g * np.dot(Q[nxt_s, :], pi_nxt) - Q[cur_s, cur_a]
            w2 = reward(new_s) + g * np.dot(Q[new_s, :], pi_new) - Q[cur_s, cur_a]
            
            G[cur_s, cur_a] -= 0.5 * w1
            G[cur_s, cur_a] -= 0.5 * w2
            for a in range(len(actions)):
                G[new_s, a] += 0.5 * pi_new[a] * g * w1
                G[nxt_s, a] += 0.5 * pi_nxt[a] * g * w2
        
        Q -= (lr / batch_size) * G
        if trueQ is not None:
            errors[k] = np.linalg.norm(Q.flatten() - trueQ.flatten())
        if P is not None:
            bellman[k] = np.linalg.norm(r + (g * P.T - np.eye(2 * N)) @ Q.flatten())

    return Q, errors, bellman


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


def MC(tol = 0.001, reps = 100000):
    Q = np.zeros((N, 2))
    for s in range(N):
        for a in range(2):
            print(f'Computing Q({s}, {a})...')
            Q[s, a] = monte_carlo(s, a, tol, reps)
    return Q


def mc_P(reps = 100000):
    P = np.zeros((2 * N, 2 * N))
    for s in range(N):
        for a in range(2):
            print(f'Estimating P, column s = {s}, a = {a}...')
            for r in range(reps):
                t = transition(s, a)
                pi_t = policy_vec(t)
                for b in range(2):
                    P[2 * t + b, 2 * s + a] += pi_t[b]
    return P / reps

###############################################################################
# Run experiment
###############################################################################
bigP = mc_P(reps = 50000)

r = np.zeros(N * 2)
for s in range(N):
    for a in range(2):
        for t in range(N):
            r[2 * s + a] += (bigP[2 * t + 0, 2 * s + a] + bigP[2 * t + 1, 2 * s + a]) * reward(t)

Q_bellman = np.linalg.solve(np.eye(N * 2) - g * bigP.T, r)
Q_bellman = Q_bellman.reshape((N, 2))
Q_actual  = Q_bellman.copy()

Q_UB, errors_UB, bellman_UB    = UB(T, trueQ = Q_bellman, Q_init = np.zeros((N, 2)), batch_size = 50, P = bigP, r = r)
Q_DS, errors_DS, bellman_DS    = DS(T, trueQ = Q_bellman, Q_init = np.zeros((N, 2)), batch_size = 50, P = bigP, r = r)
Q_BFF, errors_BFF, bellman_BFF = BFF(T, trueQ = Q_bellman, Q_init = np.zeros((N, 2)), batch_size = 50, P = bigP, r = r)

initial_error  = np.linalg.norm(np.zeros(N * 2) - Q_actual.flatten())
rel_errors_UB  = [err / initial_error for err in errors_UB]
rel_errors_DS  = [err / initial_error for err in errors_DS]
rel_errors_BFF = [err / initial_error for err in errors_BFF]

log_errors_UB = [np.log10(err) for err in rel_errors_UB]
log_errors_DS = [np.log10(err) for err in rel_errors_DS]
log_errors_BFF = [np.log10(err) for err in rel_errors_BFF]

k = min([len(log_errors_UB), len(log_errors_DS), len(log_errors_BFF)])

plt.figure()
plt.plot(log_errors_UB[:k], label='ub', color='b')
plt.plot(log_errors_DS[:k], label='ds', color='r')
plt.plot(log_errors_BFF[:k], label='bff', color='g')
plt.xlabel('Iteration')
plt.ylabel('Relative error decay (log10 scale)')
plt.title('Relative training error decay, uniform (s, a) sampling')
plt.legend()
plt.savefig('plots/1_error.png')

plt.figure()
plt.plot(bellman_UB[:k], label='ub', color='b')
plt.plot(bellman_DS[:k], label='ds', color='r')
plt.plot(bellman_BFF[:k], label='bff', color='g')
plt.xlabel('Iteration')
plt.ylabel('Norm of Bellman residual')
plt.title('Bellman residual decay, uniform (s, a) sampling')
plt.legend()
plt.savefig('plots/1_bellman.png')

Q_actual = Q_actual.reshape((N, 2))
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(range(N), Q_actual[:, 0], label='true', color='c')
plt.plot(range(N), Q_UB[:, 0], label='ub', color='b')
plt.plot(range(N), Q_DS[:, 0], label='ds', color='r')
plt.plot(range(N), Q_BFF[:, 0], label='bff', color='g')
plt.xlabel('(state, action) pair')
plt.ylabel('Q value')
plt.title('Learned Q function, action 0, uniform (s, a) sampling')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(N), Q_actual[:, 1], label='true', color='c')
plt.plot(range(N), Q_UB[:, 1], label='ub', color='b')
plt.plot(range(N), Q_DS[:, 1], label='ds', color='r')
plt.plot(range(N), Q_BFF[:, 1], label='bff', color='g')
plt.xlabel('(state, action) pair')
plt.ylabel('Q value')
plt.title('Learned Q function, action 1, uniform (s, a) sampling')
plt.legend()
plt.savefig('plots/1_q.png')