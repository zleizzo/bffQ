import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import csv
import sys


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
    return torch.Tensor([np.cos(s), np.sin(s)])


def policy_vec(s):
    """
    Computes the vector of action probabilities pi(s).
    See Section 4.2.1, line 365.
    """
    return np.array([0.5 + np.sin(s) / 5, 0.5 - np.sin(s) / 5])


def policy(s):
    """
    Returns an action based on the fixed policy at s.
    Returns 0 with probability 1/2 + sin(s) / 5.
    Returns 1 with probability 1/2 - sin(s) / 5.
    """
    if np.random.rand() <= 0.5 + np.sin(s) / 5:
        return 0
    else:
        return 1


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


def compute_j(cur_s, cur_a, nxt_s, rwd, Q):
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


def compute_grad_j(cur_s, cur_a, nxt_s, rwd, Q):
    """
    Computes the gradient of j with respect to the parameters of Q.
    """
    Q.zero_grad()
    computation = compute_j(cur_s, cur_a, nxt_s, rwd, Q)
    computation.backward()
    return [w.grad.data for w in Q.parameters()]


def compute_grad_y(cur_s, cur_a, y):
    cur_s = map_to_input(cur_s)
    y.zero_grad()
    computation = y(cur_s)[cur_a]
    computation.backward()
    return [w.grad.data for w in y.parameters()]


def compute_y_step(cur_s, cur_a, nxt_s, rwd, Q, y):
    grads  = [torch.zeros(w.shape) for w in y.parameters()]
    grad_y = compute_grad_y(cur_s, cur_a, y)

    for i in range(len(grads)):
        delta    = compute_j(cur_s, cur_a, nxt_s, rwd, Q)
        grads[i] = grad_y[i].mul(delta - y(map_to_input(cur_s))[cur_a])
    
    return grads


def compute_Q_step(cur_s, cur_a, nxt_s, rwd, Q, y):
    """
    Note: This requires y to be updated to w_{k+1} first!
    """
    grads = [torch.zeros(w.shape) for w in y.parameters()]
    grad_j = compute_grad_j(cur_s, cur_a, nxt_s, rwd, Q)
    
    for i in range(len(grads)):
        grads[i] = grad_j[i].mul(y(map_to_input(cur_s))[cur_a])
    
    return grads


def compute_batch_y_step(batch, Q, y):
    grads = [torch.zeros(w.shape) for w in Q.parameters()]
    batch_size = len(batch)
    
    for experience in batch:
        cur_s = experience[0]
        cur_a = experience[1]
        nxt_s = experience[2]
        rwd   = experience[4]
        
        minibatch_grads = compute_y_step(cur_s, cur_a, nxt_s, rwd, Q, y)
        for l in range(len(grads)):
            grads[l] += minibatch_grads[l] / batch_size
        
    return grads


def compute_batch_Q_step(batch, Q, y):
    
    grads = [torch.zeros(w.shape) for w in Q.parameters()]
    batch_size = len(batch)

    for experience in batch:
        cur_s = experience[0]
        cur_a = experience[1]
        nxt_s = experience[2]
        rwd   = experience[4]
        
        minibatch_grads = compute_Q_step(cur_s, cur_a, nxt_s, rwd, Q, y)
        for l in range(len(grads)):
            grads[l] += minibatch_grads[l] / batch_size
        
    return grads


def pd_step(batch, Q, y, beta = 0.001, eta = 0.001):
    y_steps = compute_batch_y_step(batch, Q, y)
    with torch.no_grad():
        for step, w in zip(y_steps, y.parameters()):
            w.add_(step, alpha=beta)
    
    Q_steps = compute_batch_Q_step(batch, Q, y)
    with torch.no_grad():
        for step, w in zip(Q_steps, Q.parameters()):
            w.sub_(step, alpha=eta)


def PD(T, batch_size, beta0, eta0, e = 0.1, Q = Net(), y = Net(), trueQgraph = None, adaptive = True):
    """
    Training via PD algorithm.
    
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
    
    print('Starting PD...')
    
    cur_s = np.random.rand() * 2 * np.pi
    cur_a = policy(cur_s)
    
    nxt_s = transition(cur_s, cur_a)
    nxt_a = policy(nxt_s)
    
    ftr_s = transition(nxt_s, nxt_a)
    
    for k in range(int(T / batch_size)):
        if k % 100 == 0 and k > 0:
            print(f'ETA: {round(((time.time() - start) * (int(T / batch_size) - k) / k) / 60, 2)} min')
        
        # Relationship to notation in the paper:
        #   cur_s = s_m
        #   cur_a = a_m
        #   nxt_s = s_{m+1}
        #   new_s = s'_{m+1} = s_m + (s_{m+2} - s_{m+1})
        #   ftr_s = s_{m+2}
        batch = [None] * batch_size
        for i in range(batch_size):
            cur_s = nxt_s
            cur_a = nxt_a
            
            nxt_s = ftr_s
            nxt_a = policy(nxt_s)
            new_s = nxt_s
            rwd   = reward(nxt_s)
            done  = False
            
            ftr_s = transition(nxt_s, nxt_a)
            
            batch[i] = (cur_s, cur_a, nxt_s, new_s, rwd, done)
        
        if adaptive:
            beta = beta0 * (k + 1) ** (-0.75)
            eta  = eta0 * (k + 1) ** (-0.5)
        else:
            beta = beta0
            eta  = eta0
        pd_step(batch, Q, y, beta, eta)
            
        if trueQgraph is not None:
            with torch.no_grad():
                Qgraph = Q(z)
                errors[k] = L2_error(Qgraph, trueQgraph)
                if k > 0 and errors[k] > 10 * errors[0]:
                    print('Training is diverging. Aborting.')
                    break
                
    return Q, errors


###############################################################################
# EXPERIMENTS
###############################################################################
# Define hyperparameters.
experiment_n = int(sys.argv[1])
T            = 100 # Length of training trajectory.
beta0        = 0.5
eta0         = 0.5     # Initial vals for beta and eta.
batch_size   = 50      # Batch size.
e            = 0.5     # Epsilon for epsilon-greedy choice. Note that this is not
                     # the same as the epsilon in the Markov chain dynamics.
                     # Maybe this should be 0.5 instead of 0.1?

np.random.seed(experiment_n)
torch.manual_seed(experiment_n)

# First, load optimal Q.
path = 'csvs/nn_eval/'
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

# Train Q according to each other the methods. Get values for Q as well as
# error decay during training. Also compute a Monte Carlo estimate for Q for a
# fixed e-greedy policy based on trueQ.
# Q_UB,  e_UB  = UB(T, learning_rate, batch_size, e, Net(), trueQgraph)
# Q_DS,  e_DS  = DS(T, learning_rate, batch_size, e, Net(), trueQgraph)
# Q_BFF, e_BFF = BFF(T, learning_rate, batch_size, e, Net(), trueQgraph)
Q_PD,  e_PD  = PD(T, batch_size, beta0, eta0, e, Net(), Net(), trueQgraph, adaptive = True)
# Q_MC         = MC(trueQ, e = 0, reps = 1000)

# Compute the graphs of each of the learned Q functions.
# The Monte Carlo Q is already given as a graph rather than a function; we are
# just keeping variable names consistent across methods.
# mc  = Q_MC
# ub  = Q_UB(z).detach()
# ds  = Q_DS(z).detach()
# bff = Q_BFF(z).detach()
pd  = Q_PD(z).detach()

# Graph Q(s, 0) vs. s.
plt.figure()
plt.subplot(1,2,1)
plt.plot(x, true[:, 0], label='true', color='m')
plt.plot(x, pd[:, 0], label='pd', color='c')
# plt.plot(x, mc[:, 0], label='mc', color='c')
# plt.plot(x, ub[:, 0], label='ub', color='b')
# plt.plot(x, ds[:, 0], label='ds', color='r')
# plt.plot(x, bff[:, 0], label='bff', color='g')
plt.title('Q, action 1')
plt.legend()

# Graph Q(s, 1) vs. s.
plt.subplot(1,2,2)
plt.plot(x, true[:, 1], label='true', color='m')
plt.plot(x, pd[:, 1], label='pd', color='c')
# plt.plot(x, mc[:, 1], label='mc', color='c')
# plt.plot(x, ub[:, 1], label='ub', color='b')
# plt.plot(x, ds[:, 1], label='ds', color='r')
# plt.plot(x, bff[:, 1], label='bff', color='g')
plt.title('Q, action 2')
plt.legend()
plt.savefig(f'plots/nn_q_pd_eval_{experiment_n}.png')


# Compute relative errors for each method.
# rel_e_UB  = [err / e_UB[0]  for err in e_UB]
# rel_e_DS  = [err / e_DS[0]  for err in e_DS]
# rel_e_BFF = [err / e_BFF[0] for err in e_BFF]
rel_e_PD  = [err / e_PD[0]  for err in e_PD]

# Compute log relative errors for each method.
# log_e_UB  = [np.log10(err) for err in rel_e_UB]
# log_e_DS  = [np.log10(err) for err in rel_e_DS]
# log_e_BFF = [np.log10(err) for err in rel_e_BFF]
log_e_PD  = [np.log10(err) for err in rel_e_PD]

# Plot log relative error for each method.
plt.figure()
# plt.plot(log_e_UB,  label='ub',  color='b')
# plt.plot(log_e_DS,  label='ds',  color='r')
# plt.plot(log_e_BFF, label='bff', color='g')
plt.plot(log_e_PD,  label='pd',  color='c')
plt.title('Relative error decay, log scale')
plt.legend()
plt.savefig(f'plots/nn_error_eval_{experiment_n}.png')

# Save Q data to csv files for easy re-plotting.
# with open('csvs/control/q_mc.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for row in mc:
#         writer.writerow(row)

# with open('csvs/control/q_ub.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for row in ub.numpy():
#         writer.writerow(row)

# with open('csvs/control/q_ds.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for row in ds.numpy():
#         writer.writerow(row)

# with open('csvs/control/q_bff.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for row in bff.numpy():
#         writer.writerow(row)

# with open('csvs/control/q_true.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for row in true.numpy():
#         writer.writerow(row)

with open(f'csvs/nn_eval/q_pd_{experiment_n}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in pd.numpy():
        writer.writerow(row)

# Save error data to csv files for easy re-plotting.
# with open('csvs/control/error_ub.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(log_e_UB)

# with open('csvs/control/error_ds.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(log_e_DS)

# with open('csvs/control/error_bff.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(log_e_BFF)

with open(f'csvs/nn_eval/error_pd_{experiment_n}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(log_e_PD)