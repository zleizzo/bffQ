import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import csv
import sys
from collections import deque
from copy import deepcopy


class Net(nn.Module):
    """
    Define neural net architecture.
    Input: s in R^2
    Computation:
    s_0 = s
    s_1 = W_1 * s_0 + b_1, W_1 in R^{50 x 2}, b_1 in R^{50}, s_1 in R^{50}
    a_1 = cos(s_1) (applied coordinatewise)
    s_2 = W_2 * a_1 + b_2, W_2 in R^{50 x 50}, b_2 in R^{50}, s_2 in R^{50}
    a_2 = cos(s_2) (applied coordinatewise)
    output = W_3 * a_2 + b_3, W_3 in R^{2 x 50}, b_3 in R^2, output in R^2
    
    The output of this net on input (cos(s), sin(s)) is a 2D vector whose first
    (index 0) entry is Q(s, -1) and whose second (index 1) entry is Q(s, 1).
    """
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.cos(self.fc1(x))
        x = torch.cos(self.fc2(x))
        x = self.fc3(x)
        return x

# Define Markov chain parameters
# Refer to Section 2.1 of the paper
# g = gamma
sigma    = 0.2
epsilon  = 2 * np.pi / 32
sqrt_eps = np.sqrt(epsilon)
g        = 0.9


def transition(s, a):
    """
    Computes a transition according to the Markov chain dynamics starting at
    state s and taking action a.
    Refer to Section 4.2 in the paper. a = 0 in the code corresponds to a = -1
    in the paper, while a = 1 is the same in the code and the paper.
    """
    with torch.no_grad():
        a = 2 * a - 1 # Map {0, 1} input to {-1, 1}
        delta_s = a * epsilon + sigma * np.random.normal() * sqrt_eps
        return (s + delta_s) % (2 * np.pi)


def reward(s):
    """
    The reward function for state s.
    Refer to Section 4.2 in the paper.
    """
    return np.sin(s) + 1


def map_to_input(s):
    """
    We map each s in [0, 2pi) to (cos(s), sin(s)) and use this as input.
    See Section 4.2.1 in the paper.
    """
    return torch.tensor([np.cos(s), np.sin(s)], dtype=torch.float)


def e_greedy(choices, e):
    """
    choices: List of floats. In our case, choices will be the vector Q(s) in R^2.
    e: epsilon for the epsilon-greedy algorithm.
    
    Output: argmax(choices) w.p. 1-e; uniform from among other indices (excluding
            the argmax index) w.p. e.
    """
    argmax = torch.argmax(choices) # Returns the index of the maximum value in choices
    if np.random.rand() <= e: # np.random.rand() returns a Unif[0, 1] sample.
        while True:
            i = np.random.randint(0, len(choices)) # np.random.randint(a, b) returns a uniform random integer in [a, b)
            if i != argmax: # This if statement means we choose uniformly from all indices except the argmax.
                return i    # (Since Q(s) in R^2, there is only one other choice.)
    else:
        return argmax


def L2_error(Q1graph, Q2graph):
    """
    Approximates the L2 norm of Q1 - Q2.
    
    Subdivide [0, 2pi) into [0, 2pi/n), [2pi/n, 2*2pi/n), ..., [(n-1)*2pi/n, 2pi).
    Q1graph is given by Q1graph[i] = Q1(i * 2pi/n), i = 0, ..., n. (Similarly for Q2graph.)
    
    torch.norm(Q1graph[:n] - Q2graph[:n]) = sqrt((Q1(0) - Q2(0))^2 + ... + (Q1((n-1)*2pi/n) - Q2((n-1)*2pi/n))^2)
    Multiplying by sqrt(2pi/n) gives:
        sqrt((Q1(0) - Q2(0))^2 * 2pi/n + ... + (Q1((n-1)*2pi/n) - Q2((n-1)*2pi/n))^2 * 2pi/n)
    
    This is an approximation of the L2 norm of Q1 - Q2 with the left Riemann sum and the
    subdivision of [0, 2pi) specified above.    
    """
    n = len(Q1graph) - 1
    with torch.no_grad():
        norm = torch.norm(Q1graph[:n] - Q2graph[:n])
        norm *= np.sqrt(2 * np.pi / n)
    return norm.numpy()


def compute_j(cur_s, cur_a, nxt_s, Q):
    """
    Computes the j function from Section 2.2.2, Algorithm 2.6.
    Relationship to notation in the paper:
        cur_s = s_m
        cur_a = a_m
        nxt_s = s_{m+1}
    """
    rwd   = reward(nxt_s)
    cur_s = map_to_input(cur_s)
    nxt_s = map_to_input(nxt_s)
    
    return rwd + g * torch.max(Q(nxt_s)) - Q(cur_s)[cur_a]


def compute_grad_j(cur_s, cur_a, nxt_s, Q):
    """
    Computes the gradient of j with respect to the parameters of Q.
    """
    Q.zero_grad()
    computation = compute_j(cur_s, cur_a, nxt_s, Q)
    computation.backward()
    return [w.grad.data for w in Q.parameters()]


def UB(T, learning_rate, batch_size, e = 0.1, Q = Net(), trueQgraph = None):
    """
    Unbiased SGD.
    
    We perform online, on-policy learning. That is, we use each step in the trajectory to
    train exactly once, and we generate the trajectory according to an e-greedy action choice
    based on our current approximation for Q*.
    
    T = number of points to be generated from the trajectory.
    The total number of SGD steps is therefore T / batch_size.
    
    e = epsilon for the epsilon-greedy action choice.
    
    Q = net that we will train to approximate Q*.
    
    trueQgraph = vector s.t. trueQgraph[i] = Q*(i * 2pi/n).
    Used for calculating L2 error.
    """
    start = time.time() # Used for estimating runtime.
    
    errors = np.zeros(int(T / batch_size)) # Initialize L2 error storage.
    
    # These two lines just generate the points (cos(i * 2pi/n), sin(i * 2pi/n)).
    # These get fed into Q so we can compute L2 error.
    x = np.linspace(0, 2 * np.pi)
    z = torch.stack([map_to_input(s) for s in x])
    
    print('Starting UB SGD...')
    
    # Choose a random initial state
    nxt_s = np.random.rand() * 2 * np.pi
    
    # k denotes the k-th step of SGD.
    for k in range(int(T / batch_size)):
        # Runtime estimate.
        if k % 100 == 0 and k > 0:
            print(f'ETA: {round(((time.time() - start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
        
        # Initialize an array to accumulate gradients with respect to each of Q's parameters.
        grads = [torch.zeros(w.shape) for w in Q.parameters()]
        
        # Compute the gradient on the next batch.
        # Relationship to notation in the paper:
        #   cur_s = s_m
        #   cur_a = a_m
        #   nxt_s = s_{m+1}
        #   new_s = s'_{m+1}
        for i in range(batch_size):
            # Move to next point in the trajectory and select an e-greedy action.
            cur_s = nxt_s
            cur_a = e_greedy(Q(map_to_input(cur_s)), e)
            
            # Generate the next step in the trajectory based on our current state and action.
            nxt_s = transition(cur_s, cur_a)
            
            # For UB SGD, generate an independent copy of the next state.
            new_s = transition(cur_s, cur_a)
            
            # Compute j and grad_j.
            j      = compute_j(cur_s, cur_a, nxt_s, Q)
            grad_j = compute_grad_j(cur_s, cur_a, new_s, Q)
            
            # Compute the stochastic gradient j * grad_j for each of the params 
            # and add it to the batch gradient.
            # We divide by the batch size since the batch gradient is typically computed as the
            # average gradient over the batch.
            for l in range(len(grads)):
                grads[l] += (j / batch_size) * grad_j[l]
        
        # After computing the batch gradient, update the parameters for Q.
        # w.data.sub_(learning_rate * grad) subtracts learning_rate * grad from w in place,
        # i.e. sets w = w - learning_rate * grad.
        for w, grad in zip(Q.parameters(), grads):
            w.data.sub_(learning_rate * grad)
        
        # If we already know Q*, compute the L2 error between our current Q and Q*.
        if trueQgraph is not None:
            with torch.no_grad():
                Qgraph = Q(z)
                errors[k] = L2_error(Qgraph, trueQgraph)
    
    # Finally, return the neural net we learned to approximate Q as well as our errors.
    return Q, errors


def DS(T, learning_rate, batch_size, e = 0.1, Q = Net(), trueQgraph = None):
    """
    Double sampling SGD.
    
    This is identical to UB SGD, except that new_s = nxt_s rather than an
    independent sample. This corresponds to taking s'_{m+1} = s_{m+1}.
    """
    
    start = time.time()
    
    errors = np.zeros(int(T / batch_size))
    
    x = np.linspace(0, 2 * np.pi)
    z = torch.stack([map_to_input(s) for s in x])
    
    print('Starting DS SGD...')
    
    nxt_s = np.random.rand() * 2 * np.pi
    
    for k in range(int(T / batch_size)):
        if k % 100 == 0 and k > 0:
            print(f'ETA: {round(((time.time() - start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
        
        grads = [torch.zeros(w.shape) for w in Q.parameters()]
        
        for i in range(batch_size):
            cur_s = nxt_s
            cur_a = e_greedy(Q(map_to_input(cur_s)), e)
            
            nxt_s = transition(cur_s, cur_a)
            
            # We double-sample, setting s'_{m+1} = s_{m+1}.
            new_s = nxt_s
            
            j      = compute_j(cur_s, cur_a, nxt_s, Q)
            grad_j = compute_grad_j(cur_s, cur_a, new_s, Q)
                
            for l in range(len(grads)):
                grads[l] += (j / batch_size) * grad_j[l]
                
        for w, grad in zip(Q.parameters(), grads):
            w.data.sub_(learning_rate * grad)
        
        if trueQgraph is not None:
            with torch.no_grad():
                Qgraph = Q(z)
                errors[k] = L2_error(Qgraph, trueQgraph)
    
    return Q, errors


def BFF(T, learning_rate, batch_size, e = 0.1, Q = Net(), trueQgraph = None):
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
    
    x = np.linspace(0, 2 * np.pi)
    z = torch.stack([map_to_input(s) for s in x])
    
    print('Starting BFF SGD...')
    
    cur_s = np.random.rand() * 2 * np.pi
    cur_a = e_greedy(Q(map_to_input(cur_s)), e)
    
    nxt_s = transition(cur_s, cur_a)
    nxt_a = e_greedy(Q(map_to_input(nxt_s)), e)
    
    ftr_s = transition(nxt_s, nxt_a)
    
    for k in range(int(T / batch_size)):
        if k % 100 == 0 and k > 0:
            print(f'ETA: {round(((time.time() - start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
        
        grads = [torch.zeros(w.shape) for w in Q.parameters()]
        
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
            nxt_a = e_greedy(Q(map_to_input(nxt_s)), e)
            
            ftr_s = transition(nxt_s, nxt_a)
            
            new_s = cur_s + (ftr_s - nxt_s)
            
            j      = compute_j(cur_s, cur_a, nxt_s, Q)
            grad_j = compute_grad_j(cur_s, cur_a, new_s, Q)
                
            for l in range(len(grads)):
                grads[l] += (j / batch_size) * grad_j[l]
                
        for w, grad in zip(Q.parameters(), grads):
            w.data.sub_(learning_rate * grad)
            
        if trueQgraph is not None:
            with torch.no_grad():
                Qgraph = Q(z)
                errors[k] = L2_error(Qgraph, trueQgraph)
                
    return Q, errors


def monte_carlo(s, a, trueQ, e = 0.1, tol = 0.001, reps = 100): 
    """
    Computes a Monte Carlo estimate for the Q function given a value for Q* (trueQ)
    and a fixed epsilon-greedy policy based on Q*.
    
    s, a  = Starting state, action pair.
    trueQ = Estimate for Q*.
    e     = Value of epsilon for epsilon-greedy policy.
    tol   = For each trial, we run the trajectory until the total discounted future
            reward can be no more than tol.
    reps  = Number of trials used to estimate Q(s, a).
    """
    # T is defined so that the total reward incurred from time T to infinity is
    # at most tol.
    R_max = 2 # Max single-step reward
    T = int(np.log((1 - g) * tol / R_max) / np.log(g)) + 1
    
    total = 0
    for r in range(reps):
        s_cur = s
        a_cur = a
        discount = 1
        for t in range(1, T):
            # Transition, then compute reward.
            s_cur = transition(s_cur, a_cur)
            total += reward(s_cur) * discount
            a_cur = e_greedy(trueQ(map_to_input(s_cur)), e)
            discount *= g
            
    empirical_avg = total / reps
    return empirical_avg


def MC(trueQ, e = 0.1, tol = 0.001, reps = 1000, divisions = 50):
    """
    Computes a Monte Carlo estimate for the graph of Q (evaluated with an e-greedy
    policy based on trueQ) by running the monte_carlo method above on a mesh of
    points in [0, 2pi).
    """
    Q = np.zeros((divisions, 2))
    
    for i, s in zip(range(divisions), np.linspace(0, 2 * np.pi, divisions)):
        for a in range(2):
            print(f'Computing Q({s}, {a})...')
            Q[i, a] = monte_carlo(s, a, trueQ, e, tol, reps)
            
    return Q


def nBFF(n, T, learning_rate, batch_size, e = 0.5, Q = Net(), trueQgraph = None):
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
    
    x = np.linspace(0, 2 * np.pi)
    z = torch.stack([map_to_input(s) for s in x])
    
    print('Beginning nBFF...')
    experience_buffer = deque(maxlen = n + 2)
    
    cur_s = np.random.rand() * 2 * np.pi
    cur_a = e_greedy(Q(map_to_input(cur_s)), e)
    experience_buffer.append((cur_s, cur_a))
    
    for i in range(n + 1):
        cur_s = experience_buffer[i][0]
        cur_a = experience_buffer[i][1]
        
        nxt_s = transition(cur_s, cur_a)
        nxt_a = e_greedy(Q(map_to_input(nxt_s)), e)
        experience_buffer.append((nxt_s, nxt_a))
    
    for k in range(int(T / batch_size)):
        if k % 100 == 0 and k > 0:
            print(f'ETA: {round(((time.time() - start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
        
        grads = [torch.zeros(w.shape) for w in Q.parameters()]
        for i in range(batch_size):
            cur_s = experience_buffer[0][0]
            cur_a = experience_buffer[0][1]
            
            nxt_s = experience_buffer[1][0]
            nxt_a = experience_buffer[1][1]
            
            j          = compute_j(cur_s, cur_a, nxt_s, Q)
            sum_grad_j = [torch.zeros(w.shape) for w in Q.parameters()]
            for m in range(n):
                ds    = experience_buffer[m + 2][0] - experience_buffer[m + 1][0] # Compute \Delta s
                new_s = cur_s + ds
                
                grad_j = compute_grad_j(cur_s, cur_a, new_s, Q)
                for l in range(len(grad_j)):
                    sum_grad_j[l] += grad_j[l]
            
            for l in range(len(grads)):
                grads[l] += (j / (batch_size * n)) * sum_grad_j[l]
            
            ftr_s = transition(experience_buffer[-1][0], experience_buffer[-1][1])
            ftr_a = e_greedy(Q(map_to_input(ftr_s)), e)
            experience_buffer.append((ftr_s, ftr_a))
                
        for w, grad in zip(Q.parameters(), grads):
            w.data.sub_(learning_rate * grad)
            
        if trueQgraph is not None:
            with torch.no_grad():
                Qgraph = Q(z)
                errors[k] = L2_error(Qgraph, trueQgraph)
    
    return Q, errors

###############################################################################
# EXPERIMENTS
###############################################################################
seed = int(sys.argv[1])
# seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# Define hyperparameters.
T             = 1000 # Length of training trajectory.
learning_rate = 0.1     # Learning rate.
batch_size    = 50      # Batch size.
e             = 0.5     # Epsilon for epsilon-greedy choice. Note that this is not
                        # the same as the epsilon in the Markov chain dynamics.
                        # Maybe this should be 0.5 instead of 0.1?

# # Run this section first time only to compute optimal Q.
# # First, learn true optimal Q based on a longer trajectory.
# trueQ, _ = UB(3 * T, learning_rate, batch_size, e, Net())

# # Define grid of points on which to evaluate each of the Q functions we learn
# # so that we can graph them.
# x = np.linspace(0, 2 * np.pi)
# z = torch.stack([map_to_input(s) for s in x])

# # Compute the graph of trueQ.
# trueQgraph = trueQ(z).detach()

# If optimal Q has already been computed, use this instead to save time.
path = 'csvs/control/'
true = np.zeros((50, 2))
with open(path + 'q_true.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row, i in zip(reader, range(50)):
        true[i, :] = row

# Define grid of points on which to evaluate each of the Q functions we learn
# so that we can graph them.
x = np.linspace(0, 2 * np.pi)
z = torch.stack([map_to_input(s) for s in x])

# Compute the graph of trueQ.
trueQgraph = true

init = Net()
# Train Q according to each other the methods. Get values for Q as well as
# error decay during training. Also compute a Monte Carlo estimate for Q for a
# fixed e-greedy policy based on trueQ.
Q_UB,  e_UB  = UB(T, learning_rate, batch_size, e, deepcopy(init), trueQgraph)
Q_DS,  e_DS  = DS(T, learning_rate, batch_size, e, deepcopy(init), trueQgraph)
Q_BFF, e_BFF = BFF(T, learning_rate, batch_size, e, deepcopy(init), trueQgraph)
Q_3BFF, e_3BFF = nBFF(3, T, learning_rate, batch_size, e, deepcopy(init), trueQgraph)
# Q_MC         = MC(trueQ, e = 0, reps = 1000)

# Compute the graphs of each of the learned Q functions.
# The Monte Carlo Q is already given as a graph rather than a function; we are
# just keeping variable names consistent across methods.
# mc  = Q_MC
ub  = Q_UB(z).detach()
ds  = Q_DS(z).detach()
bff = Q_BFF(z).detach()
nbff = Q_3BFF(z).detach()
# true = trueQ(z).detach()

# Graph Q(s, 0) vs. s.
plt.figure()
plt.subplot(1,2,1)
plt.plot(x, true[:, 0], label='true', color='m')
# plt.plot(x, mc[:, 0], label='mc', color='c')
plt.plot(x, ub[:, 0], label='ub', color='b')
plt.plot(x, ds[:, 0], label='ds', color='r')
plt.plot(x, bff[:, 0], label='bff', color='g')
plt.plot(x, nbff[:, 0], label='3bff', color='c')
plt.title('Q, action 1')
plt.legend()

# Graph Q(s, 1) vs. s.
plt.subplot(1,2,2)
plt.plot(x, true[:, 1], label='true', color='m')
# plt.plot(x, mc[:, 1], label='mc', color='c')
plt.plot(x, ub[:, 1], label='ub', color='b')
plt.plot(x, ds[:, 1], label='ds', color='r')
plt.plot(x, bff[:, 1], label='bff', color='g')
plt.plot(x, nbff[:, 1], label='3bff', color='c')
plt.title('Q, action 2')
plt.legend()
# plt.savefig('plots/nn_q_mc_control.png')


# # Q plots without MC.
# plt.figure()
# plt.subplot(1,2,1)
# plt.plot(x, true[:, 0], label='true', color='m')
# plt.plot(x, ub[:, 0], label='ub', color='b')
# plt.plot(x, ds[:, 0], label='ds', color='r')
# plt.plot(x, bff[:, 0], label='bff', color='g')
# plt.title('Q, action 1')
# plt.legend()

# plt.subplot(1,2,2)
# plt.plot(x, true[:, 1], label='true', color='m')
# plt.plot(x, ub[:, 1], label='ub', color='b')
# plt.plot(x, ds[:, 1], label='ds', color='r')
# plt.plot(x, bff[:, 1], label='bff', color='g')
# plt.title('Q, action 2')
# plt.legend()
# # plt.savefig('plots/nn_q_control.png')

# Compute relative errors for each method.
rel_e_UB  = [err / e_UB[0]  for err in e_UB]
rel_e_DS  = [err / e_DS[0]  for err in e_DS]
rel_e_BFF = [err / e_BFF[0] for err in e_BFF]
rel_e_3BFF = [err / e_3BFF[0] for err in e_3BFF]

# Compute log relative errors for each method.
log_e_UB  = [np.log10(err) for err in rel_e_UB]
log_e_DS  = [np.log10(err) for err in rel_e_DS]
log_e_BFF = [np.log10(err) for err in rel_e_BFF]
log_e_3BFF = [np.log10(err) for err in rel_e_3BFF]

# Plot log relative error for each method.
plt.figure()
plt.plot(log_e_UB,  label='ub',  color='b')
plt.plot(log_e_DS,  label='ds',  color='r')
plt.plot(log_e_BFF, label='bff', color='g')
plt.plot(log_e_3BFF, label='3bff', color='c')
plt.title('Relative error decay, log scale')
plt.legend()
# plt.savefig('plots/nn_error_control.png')

# Save Q data to csv files for easy re-plotting.
# with open('csvs/control/q_mc.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for row in mc:
#         writer.writerow(row)

with open(f'csvs/control/q_ub_{seed}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in ub.numpy():
        writer.writerow(row)

with open(f'csvs/control/q_ds_{seed}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in ds.numpy():
        writer.writerow(row)

with open(f'csvs/control/q_1bff_{seed}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in bff.numpy():
        writer.writerow(row)

with open(f'csvs/control/q_3bff_{seed}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in nbff.numpy():
        writer.writerow(row)

# with open('csvs/control/q_true.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for row in true.numpy():
#         writer.writerow(row)

# Save error data to csv files for easy re-plotting.
with open(f'csvs/control/error_ub_{seed}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(log_e_UB)

with open(f'csvs/control/error_ds_{seed}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(log_e_DS)

with open(f'csvs/control/error_1bff_{seed}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(log_e_BFF)

with open(f'csvs/control/error_3bff_{seed}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(log_e_3BFF)