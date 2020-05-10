import gym
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func

env_name = "CartPole-v0"
env = gym.make(env_name)

###############################################################################
# Parameters
###############################################################################
g = 0.97 # Reward discount factor


###############################################################################
# Define NN architecture
###############################################################################
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 100)
#         self.fc2 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(100, env.action_space.n)

    def forward(self, x):
        x = func.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x


def e_greedy(choices, e):
    """
    Returns the index for an e-greedy choice from choices
    """
    return torch.argmax(choices) if np.random.rand() > e else np.random.randint(0, len(choices))


def compute_j(cur_s, cur_a, nxt_s, rwd, done, Q):
    cur_s = torch.Tensor(cur_s)
    nxt_s = torch.Tensor(nxt_s)
    
    best_Q = 0 if done else torch.max(Q(nxt_s))
    
    return rwd + g * best_Q - Q(cur_s)[cur_a]


def compute_batch_loss(batch, Q):
    loss = 0
    for experience in batch:
        batch_cur_s = experience[0]
        batch_cur_a = experience[1]
        batch_nxt_s = experience[2]
        batch_rwd = experience[4]
        batch_done = experience[5]
        
        loss += compute_j(batch_cur_s, batch_cur_a, batch_nxt_s, batch_rwd, batch_done, Q) ** 2
    return loss


def compute_grad_j(cur_s, cur_a, nxt_s, rwd, done, Q):
    Q.zero_grad()
    computation = compute_j(cur_s, cur_a, nxt_s, rwd, done, Q)
    computation.backward()
    return [w.grad.data for w in Q.parameters()]


def compute_F(cur_s, cur_a, nxt_s, new_s, rwd, done, Q):
    j      = compute_j(cur_s, cur_a, nxt_s, rwd, done, Q)
    grad_j = compute_grad_j(cur_s, cur_a, new_s, rwd, done, Q)
    
    grads = [None for w in Q.parameters()]
    for l in range(len(grads)):
        grads[l] = j * grad_j[l]
    
    return grads


def compute_batch_F(batch, Q):
    grads = [torch.zeros(w.shape) for w in Q.parameters()]
    batch_size = len(batch)
    
    for experience in batch:
        cur_s = experience[0]
        cur_a = experience[1]
        nxt_s = experience[2]
        new_s = experience[3]
        rwd = experience[4]
        done = experience[5]
        
        minibatch_grads = compute_F(cur_s, cur_a, nxt_s, new_s, rwd, done, Q)
        for l in range(len(grads)):
            grads[l] += minibatch_grads[l] / batch_size
        
    return grads


def adam(batch, Q, ms, vs, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    bias_correction1 = 1 - beta1 ** t
    bias_correction2 = 1 - beta2 ** t
    
    grads = compute_batch_F(batch, Q)

    # Decay the first and second moment running average coefficient
    for grad, exp_avg, exp_avg_sq, w in zip(grads, ms, vs, Q.parameters()):
        with torch.no_grad():
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1
        
            w.addcdiv_(exp_avg, denom, value=-step_size)


def my_adam(batch, Q, ms, vs, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    old = [w.clone() for w in Q.parameters()]
    adam(batch, Q, ms, vs, t, lr, beta1, beta2, eps)
    steps = [w_old - w_new for w_old, w_new in zip(old, Q.parameters())]
                
    return steps


def torch_adam(batch, optimizer, Q):
    old = [w.clone() for w in Q.parameters()]
    
    batch_size = len(batch)
    loss = compute_batch_loss(batch, Q) / (2 * batch_size)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    steps = [w_old - w_new for w_old, w_new in zip(old, Q.parameters())]
                
    return steps