import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import sys
from collections import deque

###############################################################################
# Define hyperparameters
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

np.random.seed(0)
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
def UB(T, trueQ = None, Q_init = np.zeros((N, 2)), batch_size = 1, P = None, r = None):
    """
    Unbiased SGD.
    
    We perform online, on-policy learning. That is, we use each step in the trajectory to
    train exactly once, and we generate the trajectory with actions drawn from the fixed
    policy pi.
    
    T = number of points to be generated from the trajectory.
    The total number of SGD steps is therefore T / batch_size.
    
    Q_init = initial value for the |S| x |A| matrix Q
    
    trueQ = true Q matrix, solved for exactly using the Bellman equation
    
    P = transition matrix for the Markov dynamics with fixed policy pi
    Note that this is the transition matrix on (s, a) pairs, so it is an |S||A| x |S||A| matrix.
    
    r = reward vector indexed by (s, a) pairs.
    r(s, a) = expected reward at next state starting from s with action a
    """
    start = time.time() # Used for estimating runtime.
    
    errors  = np.zeros(int(T / batch_size)) # Initialize L2 error storage.
    bellman = np.zeros(int(T / batch_size)) # Initialize Bellman residual storage.
    
    Q = Q_init.copy() # Used to create a new instance of Q, rather than modifying the Q which is supplied as an argument.

    print('Starting UB SGD...')
    
    # Choose a random initial state
    nxt_s = np.random.randint(0, N)
    
    # k denotes the k-th step of SGD.
    for k in range(int(T / batch_size)):
        # Runtime estimate.
        if k % 100 == 0 and k > 0:
            print(f'ETA: {round(((time.time() - start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
        
        # Initialize a variable to accumulate the batch gradient.
        G = np.zeros((N, 2))
        
        # Compute the gradient on the next batch.
        # Relationship to notation in the paper:
        #   cur_s = s_m
        #   cur_a = a_m
        #   nxt_s = s_{m+1}
        #   new_s = s'_{m+1}
        for i in range(batch_size):
            # Move to next point in the trajectory and select an action from the fixed policy.
            cur_s = nxt_s
            cur_a = policy(cur_s)
            
            # Generate the next step in the trajectory based on our current state and action.
            nxt_s = transition(cur_s, cur_a)
            
            # For UB SGD, generate an independent copy of the next state.
            new_s = transition(cur_s, cur_a)
            
            # Compute the policy vectors at s_{m+1} and s'_{m+1}.
            pi_nxt = policy_vec(nxt_s)
            pi_new = policy_vec(new_s)
            
            # Refer to equations (23) and (24) from the paper.
            # Here we compute the gradient for our current (state, action) and add it to
            # the batch stochastic gradient G.
            j = reward(nxt_s) + g * np.dot(Q[nxt_s, :], pi_nxt) - Q[cur_s, cur_a]
            G[cur_s, cur_a] -= j
            for a in range(len(actions)):
                G[new_s, a] += pi_new[a] * g * j
        
        # Update Q based on the batch stochastic gradient G.
        Q -= (lr / batch_size) * G
        
        # If we know the true value of Q already, compute the L2 error.
        if trueQ is not None:
            errors[k] = np.linalg.norm(Q.flatten() - trueQ.flatten())
            
        # If we know the transition matrix P, also compute the Bellman residual.
        if P is not None:
            bellman[k] = np.linalg.norm(r + (g * P.T - np.eye(2 * N)) @ Q.flatten())
    
    # Finally, return the learned Q, the L2 errors, and the Bellman residuals.
    return Q, errors, bellman


def DS(T, trueQ = None, Q_init = np.zeros((N, 2)), batch_size=1, P = None, r = None):
    """
    Double sampling SGD.
    
    This is identical to UB SGD, except that new_s = nxt_s rather than an
    independent sample. This corresponds to taking s'_{m+1} = s_{m+1}.
    """
    start = time.time()
    
    errors  = np.zeros(int(T / batch_size))
    bellman = np.zeros(int(T / batch_size))
    
    Q = Q_init.copy()

    print('Starting DS SGD...')
    
    nxt_s = np.random.randint(0, N)
    
    for k in range(int(T / batch_size)):
        if k % 100 == 0 and k > 0:
            print(f'ETA: {round(((time.time() - start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
        
        G = np.zeros((N, 2))
        
        for i in range(batch_size):
            cur_s = nxt_s
            cur_a = policy(cur_s)
            
            nxt_s = transition(cur_s, cur_a)
            
            # We double-sample, setting s'_{m+1} = s_{m+1}.
            new_s = nxt_s
            
            pi_nxt = policy_vec(nxt_s)
            pi_new = policy_vec(new_s)
            
            j = reward(nxt_s) + g * np.dot(Q[nxt_s, :], pi_nxt) - Q[cur_s, cur_a]
            G[cur_s, cur_a] -= j
            for a in range(len(actions)):
                G[new_s, a] += pi_new[a] * g * j
        
        Q -= (lr / batch_size) * G
        
        if trueQ is not None:
            errors[k] = np.linalg.norm(Q.flatten() - trueQ.flatten())
            
        if P is not None:
            bellman[k] = np.linalg.norm(r + (g * P.T - np.eye(2 * N)) @ Q.flatten())

    return Q, errors, bellman


def BFF(T, trueQ = None, Q_init = np.zeros((N, 2)), batch_size=1, P = None, r = None):
    """
    SGD with the BFF approximation.
    
    For the previous two algorithms, we only needed to keep track of the current
    time step and one time step in the future (cur_s and nxt_s, respectively).
    For BFF, we need to keep track of one additional time step. This is the main
    distinction between this method and the other two; the rest is identical
    except for the definition of s_new.
    """
    start = time.time()
    
    errors  = np.zeros(int(T / batch_size))
    bellman = np.zeros(int(T / batch_size))
    
    Q = Q_init.copy()

    print('Starting BFF SGD...')
    
    cur_s = np.random.rand() * 2 * np.pi
    cur_a = policy(cur_s)
    
    nxt_s = transition(cur_s, cur_a)
    nxt_a = policy(nxt_s)
    
    ftr_s = transition(nxt_s, nxt_a)
    
    for k in range(int(T / batch_size)):
        if k % 100 == 0 and k > 0:
            print(f'ETA: {round(((time.time() - start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
        
        G = np.zeros((N, 2))
        
        # Relationship to notation in the paper:
        #   cur_s = s_m
        #   cur_a = a_m
        #   nxt_s = s_{m+1}
        #   new_s = s'_{m+1} = s_m + (s_{m+2} - s_{m+1})
        #   ftr_s = s_{m+2}
        for i in range(batch_size):
            cur_s = nxt_s
            cur_a = nxt_a
            
            nxt_s = ftr_s
            nxt_a = policy(nxt_s)
            
            ftr_s = transition(nxt_s, nxt_a)
            
            new_s = (cur_s + (ftr_s - nxt_s)) % N
            
            pi_nxt = policy_vec(nxt_s)
            pi_new = policy_vec(new_s)
            
            j = reward(nxt_s) + g * np.dot(Q[nxt_s, :], pi_nxt) - Q[cur_s, cur_a]
            G[cur_s, cur_a] -= j
            for a in range(len(actions)):
                G[new_s, a] += pi_new[a] * g * j
        
        Q -= (lr / batch_size) * G
        
        if trueQ is not None:
            errors[k] = np.linalg.norm(Q.flatten() - trueQ.flatten())
            
        if P is not None:
            bellman[k] = np.linalg.norm(r + (g * P.T - np.eye(2 * N)) @ Q.flatten())

    return Q, errors, bellman


def nBFF(n, T, trueQ = None, Q_init = np.zeros((N, 2)), batch_size = 50, lr = 0.5):
    """
    SGD with the BFF approximation.
    
    For the previous two algorithms, we only needed to keep track of the current
    time step and one time step in the future (cur_s and nxt_s, respectively).
    For BFF, we need to keep track of one additional time step. This is the main
    distinction between this method and the other two; the rest is identical
    except for the definition of s_new.
    """
    start = time.time()
    
    errors = np.zeros(int(T / batch_size))
    
    Q = Q_init.copy()
    
    print('Beginning nBFF...')
    experience_buffer = deque(maxlen = n + 2)
    
    cur_s = np.random.randint(0, N)
    cur_a = policy(cur_s)
    experience_buffer.append((cur_s, cur_a))
    
    for i in range(n + 1):
        cur_s = experience_buffer[i][0]
        cur_a = experience_buffer[i][1]
        
        nxt_s = transition(cur_s, cur_a)
        nxt_a = policy(nxt_s)
        experience_buffer.append((nxt_s, nxt_a))
    
    for k in range(int(T / batch_size)):
        if k % 100 == 0 and k > 0:
            print(f'ETA: {round(((time.time() - start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
        
        G = np.zeros((N, 2))
        for i in range(batch_size):
            cur_s = experience_buffer[0][0]
            cur_a = experience_buffer[0][1]
            
            nxt_s = experience_buffer[1][0]
            nxt_a = experience_buffer[1][1]
            
            pi_nxt = policy_vec(nxt_s)
            j      = reward(nxt_s) + g * np.dot(Q[nxt_s, :], pi_nxt) - Q[cur_s, cur_a]
            
            G[cur_s, cur_a] -= j
            
            for m in range(n):
                ds    = experience_buffer[m + 2][0] - experience_buffer[m + 1][0] # Compute \Delta s
                new_s = (cur_s + ds) % N
                
                pi_new = policy_vec(new_s)
                for a in range(len(actions)):
                    G[new_s, a] += pi_new[a] * g * j / n
            
            ftr_s = transition(experience_buffer[-1][0], experience_buffer[-1][1])
            ftr_a = policy(ftr_s)
            experience_buffer.append((ftr_s, ftr_a))
        
        Q -= (lr / batch_size) * G
        
        if trueQ is not None:
            errors[k] = np.linalg.norm(Q.flatten() - trueQ.flatten())

    return Q, errors


def monte_carlo(s, a, tol = 0.001, reps = 1000):
    """
    Computes a Monte Carlo estimate for the Q function based on the fixed policy pi.
    
    s, a  = Starting state, action pair.
    tol   = For each trial, we run the trajectory until the total discounted future
            reward can be no more than tol.
    reps  = Number of trials used to estimate Q(s, a).
    """
    # T is defined so that the total reward incurred from time T to infinity is
    # at most tol.
    R_max = 2 # Max single-step reward.
    T = int(np.log((1 - g) * tol / R_max) / np.log(g)) + 1
    
    total = 0
    for r in range(reps):
        # Transition, then compute reward.
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
    """
    Computes a Monte Carlo estimate for the graph of Q based on the fixed policy
    pi by running the monte_carlo method above on a mesh of points in [0, 2pi).
    """
    Q = np.zeros((N, 2))
    for s in range(N):
        for a in range(2):
            print(f'Computing Q({s}, {a})...')
            Q[s, a] = monte_carlo(s, a, tol, reps)
    return Q


def mc_P(reps = 100000):
    """
    Computes a Monte Carlo estimate for the (s, a) transition matrix P with the
    fixed policy pi.
    That is, computes P in R^{|S||A| x |S||A|} such that
        P[2 * t + b, 2 * s + a] = Prob(move to (t, b) from (s, a)).
    """
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
# seed = 0
seed = int(sys.argv[1])
np.random.seed(seed)
# Compute the transition matrix.
# I called it bigP since this is the transition matrix for (s, a) pairs rather
# than just the state transition matrix.
# bigP = mc_P(reps = 50000)

# # Compute the reward vector. This is also indexed by (s, a) pairs, r in R^{|S||A|}.
# r = np.zeros(N * 2)
# for s in range(N):
#     for a in range(2):
#         for t in range(N):
#             r[2 * s + a] += (bigP[2 * t + 0, 2 * s + a] + bigP[2 * t + 1, 2 * s + a]) * reward(t)

# # Learn the true Q function by solving the Bellman equation exactly.
# Q_bellman = np.linalg.solve(np.eye(N * 2) - g * bigP.T, r)
# Q_bellman = Q_bellman.reshape((N, 2))
# Q_actual  = Q_bellman.copy()

# Compute a Monte Carlo estimate for Q. We don't really need this since we get a better
# result from solving the Bellman equation (our estimate for the transition matrix is very stable).
#Q_MC  = MC(tol = 0.001, reps = 1000)

# Import the true Q matrix if we've already computed it.
Q_actual = np.zeros((N, 2))
with open('csvs/tab_eval/q_true.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row, i in zip(reader, range(50)):
        Q_actual[i, :] = row

# # Compute Q according to each of the training methods. Also get the L2 errors and Bellman residuals from training.
# Q_UB, errors_UB, bellman_UB    = UB(T, trueQ = Q_bellman, Q_init = np.zeros((N, 2)), batch_size = 50, P = bigP, r = r)
# Q_DS, errors_DS, bellman_DS    = DS(T, trueQ = Q_bellman, Q_init = np.zeros((N, 2)), batch_size = 50, P = bigP, r = r)
# Q_BFF, errors_BFF, bellman_BFF = BFF(T, trueQ = Q_bellman, Q_init = np.zeros((N, 2)), batch_size = 50, P = bigP, r = r)

Q_UB, errors_UB, bellman_UB    = UB(T, trueQ = Q_actual, Q_init = np.zeros((N, 2)), batch_size = 50, P = None, r = None)
Q_DS, errors_DS, bellman_DS    = DS(T, trueQ = Q_actual, Q_init = np.zeros((N, 2)), batch_size = 50, P = None, r = None)
Q_BFF, errors_BFF, bellman_BFF = BFF(T, trueQ = Q_actual, Q_init = np.zeros((N, 2)), batch_size = 50, P = None, r = None)
Q_nBFF, errors_nBFF = nBFF(5, T, trueQ = Q_actual, Q_init = np.zeros((N, 2)), batch_size = 50, lr = lr)

# Compute the relative L2 errors of each method.
initial_error  = np.linalg.norm(np.zeros(N * 2) - Q_actual.flatten())
rel_errors_UB  = [err / initial_error for err in errors_UB]
rel_errors_DS  = [err / initial_error for err in errors_DS]
rel_errors_BFF = [err / initial_error for err in errors_BFF]
rel_errors_nBFF = [err / initial_error for err in errors_nBFF]

# Compute log relative L2 errors of each method.
log_errors_UB = [np.log10(err) for err in rel_errors_UB]
log_errors_DS = [np.log10(err) for err in rel_errors_DS]
log_errors_BFF = [np.log10(err) for err in rel_errors_BFF]
log_errors_nBFF = [np.log10(err) for err in rel_errors_nBFF]

# Plot log relative L2 errors.
plt.figure()
plt.plot(log_errors_UB, label='ub', color='b')
plt.plot(log_errors_DS, label='ds', color='r')
plt.plot(log_errors_BFF, label='1bff', color='g')
plt.plot(log_errors_nBFF, label='5bff', color='c')
plt.title('Relative error decay, log scale')
plt.legend()
plt.savefig('plots/tabular_error.png')

# Plot Bellman residuals.
# plt.figure()
# plt.plot(bellman_UB, label='ub', color='b')
# plt.plot(bellman_DS, label='ds', color='r')
# plt.plot(bellman_BFF, label='bff', color='g')
# plt.title('Bellman residual decay')
# plt.legend()
# plt.savefig('plots/tabular_bellman.png')

# Plot learned Q.
Q_actual = Q_actual.reshape((N, 2))
plt.figure()
plt.subplot(1, 2, 1)
#plt.plot(range(N), Q_MC[:, 0], label='mc', color='m')
plt.plot(Q_actual[:, 0], label='true', color='c')
plt.plot(Q_UB[:, 0], label='ub', color='b')
plt.plot(Q_DS[:, 0], label='ds', color='r')
plt.plot(Q_BFF[:, 0], label='1bff', color='g')
plt.plot(Q_nBFF[:, 0], label='5bff', color='c')
plt.title('Q, action 1')
plt.legend()

plt.subplot(1, 2, 2)
#plt.plot(range(N), Q_MC[:, 1], label='mc', color='m')
plt.plot(Q_actual[:, 1], label='true', color='c')
plt.plot(Q_UB[:, 1], label='ub', color='b')
plt.plot(Q_DS[:, 1], label='ds', color='r')
plt.plot(Q_BFF[:, 1], label='1bff', color='g')
plt.plot(Q_nBFF[:, 1], label='5bff', color='c')
plt.title('Q, action 2')
plt.legend()
# plt.savefig('plots/tabular_q.png')


# Save Q data to csv files for easy re-plotting.
with open(f'csvs/tab_eval/q_ub_{seed}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in Q_UB:
        writer.writerow(row)

with open(f'csvs/tab_eval/q_ds_{seed}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in Q_DS:
        writer.writerow(row)

with open(f'csvs/tab_eval/q_bff_{seed}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in Q_BFF:
        writer.writerow(row)

with open(f'csvs/tab_eval/q_5bff_{seed}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in Q_nBFF:
        writer.writerow(row)
# with open('csvs/tab_eval/q_true.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for row in Q_actual:
#         writer.writerow(row)

# Save error data to csv files for easy re-plotting.
with open(f'csvs/tab_eval/error_ub_{seed}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(log_errors_UB)

with open(f'csvs/tab_eval/error_ds_{seed}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(log_errors_DS)

with open(f'csvs/tab_eval/error_bff_{seed}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(log_errors_BFF)

with open(f'csvs/tab_eval/error_5bff_{seed}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(log_errors_nBFF)