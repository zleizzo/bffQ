import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import sys
from collections import deque

###############################################################################
# Parameters for dynamics
###############################################################################
g          = 0.9
N          = 32
sigma      = 1
actions    = [2 * np.pi / N, -2 * np.pi / N]
grid_size  = 2 * np.pi / N
dt         = 1

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
def UB(T, trueQ = None, Q_init = np.zeros((N, 2)), batch_size = 1, lr = 0.1, P = None, r = None):
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
            
            # Refer to equations (23) and (24) from the paper.
            # Here we compute the gradient for our current (state, action) and add it to
            # the batch stochastic gradient G.
            j = reward(nxt_s) + g * max(Q[nxt_s, :]) - Q[cur_s, cur_a]
            G[cur_s, cur_a] -= j
            max_a = np.argmax(Q[new_s, :])
            G[new_s, max_a] += g * j
        
        # Update Q based on the batch stochastic gradient G.
        Q -= (lr / batch_size) * G
        
        # If we know the true value of Q already, compute the L2 error.
        if trueQ is not None:
            errors[k] = np.linalg.norm(Q.flatten() - trueQ.flatten())
            
        # # If we know the transition matrix P, also compute the Bellman residual.
        # if P is not None:
        #     bellman[k] = np.linalg.norm(r + (g * P.T - np.eye(2 * N)) @ Q.flatten())
    
    # Finally, return the learned Q, the L2 errors, and the Bellman residuals.
    return Q, errors#, bellman


def DS(T, trueQ = None, Q_init = np.zeros((N, 2)), batch_size=1, lr = 0.1, P = None, r = None):
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
            
            j = reward(nxt_s) + g * max(Q[nxt_s, :]) - Q[cur_s, cur_a]
            G[cur_s, cur_a] -= j
            max_a = np.argmax(Q[new_s, :])
            G[new_s, max_a] += g * j
        
        Q -= (lr / batch_size) * G
        
        if trueQ is not None:
            errors[k] = np.linalg.norm(Q.flatten() - trueQ.flatten())
            
        if P is not None:
            bellman[k] = np.linalg.norm(r + (g * P.T - np.eye(2 * N)) @ Q.flatten())

    return Q, errors#, bellman


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
            
            j = reward(nxt_s) + g * max(Q[nxt_s, :]) - Q[cur_s, cur_a]
            
            G[cur_s, cur_a] -= j
            
            for m in range(n):
                ds    = experience_buffer[m + 2][0] - experience_buffer[m + 1][0] # Compute \Delta s
                new_s = (cur_s + ds) % N
                
                max_a = np.argmax(Q[new_s, :])
                G[new_s, max_a] += g * j / n
            
            ftr_s = transition(experience_buffer[-1][0], experience_buffer[-1][1])
            ftr_a = policy(ftr_s)
            experience_buffer.append((ftr_s, ftr_a))
        
        Q -= (lr / batch_size) * G
        
        if trueQ is not None:
            errors[k] = np.linalg.norm(Q.flatten() - trueQ.flatten())

    return Q, errors


###############################################################################
# Run experiment
###############################################################################
np.random.seed(0)

T          = 100000
lr         = 0.5
batch_size = 1000
# n          = int(sys.argv[1])


# path = 'csvs/tab_ctrl/'
# true = np.zeros((N, 2))
# with open(path + 'q_true.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',')
#     for row, i in zip(reader, range(50)):
#         true[i, :] = row

trueQ, _ = UB(T, batch_size = batch_size, lr = lr)

# Train Q according to each other the methods. Get values for Q as well as
# error decay during training. Also compute a Monte Carlo estimate for Q.
# Q_nBFF, e_nBFF = nBFF(n, T, trueQ = true, Q_init = np.zeros((N, 2)), batch_size = batch_size, lr = lr)

# Compute the graphs of each of the learned Q functions.
# The Monte Carlo Q is already given as a graph rather than a function; we are
# just keeping variable names consistent across methods.
# nbff = Q_nBFF

# x = range(N)
# # Graph Q(s, 0) vs. s.
# plt.figure()
# plt.subplot(1,2,1)
# plt.plot(x, true[:, 0], label='true', color='m')
# plt.plot(x, nbff[:, 0], label=f'{n}bff', color='g')
# plt.title('Q, action 1')
# plt.legend()

# # Graph Q(s, 1) vs. s.
# plt.subplot(1,2,2)
# plt.plot(x, true[:, 1], label='true', color='m')
# plt.plot(x, nbff[:, 1], label=f'{n}bff', color='g')
# plt.title('Q, action 2')
# plt.legend()
# plt.savefig(f'plots/tab_q_{n}bff.png')


# # Compute relative errors for each method.
# rel_e_nBFF = [err / e_nBFF[0] for err in e_nBFF]

# # Compute log relative errors for each method.
# log_e_nBFF = [np.log10(err) for err in rel_e_nBFF]

# # Plot log relative error for each method.
# plt.figure()
# plt.plot(log_e_nBFF, label=f'{n}bff', color='g')
# plt.title('Relative error decay, log scale')
# plt.legend()
# plt.savefig(f'plots/tab_error_{n}bff.png')


with open(f'csvs/tab_ctrl/q_true.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in trueQ:
        writer.writerow(row)

# with open(f'csvs/tab_eval/error_{n}bff.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(log_e_nBFF)