# Plotting
import csv
import matplotlib.pyplot as plt
import numpy as np
    
path = 'csvs/control/'

bff = np.zeros((50, 2))
with open(path + 'q_bff.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row, i in zip(reader, range(50)):
        bff[i, :] = row

ds = np.zeros((50, 2))
with open(path + 'q_ds.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row, i in zip(reader, range(50)):
        ds[i, :] = row

ub = np.zeros((50, 2))
with open(path + 'q_ub.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row, i in zip(reader, range(50)):
        ub[i, :] = row

true = np.zeros((50, 2))
with open(path + 'q_true.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row, i in zip(reader, range(50)):
        true[i, :] = row

mc = np.zeros((50, 2))
with open(path + 'q_mc.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row, i in zip(reader, range(50)):
        mc[i, :] = row

with open(path + 'error_bff.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        e_bff = np.array(row)

with open(path + 'error_ds.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        e_ds = np.array(row)

with open(path + 'error_ub.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        e_ub = np.array(row)




plt.figure()
plt.plot(e_ub,  label='ub',  color='b')
plt.plot(e_ds,  label='ds',  color='r')
plt.plot(e_bff, label='bff', color='g')
plt.title('Relative error decay, log scale')
plt.legend()



x = np.linspace(0, 2 * np.pi)
plt.figure()
plt.subplot(1,2,1)
plt.plot(x, true[:, 0], label='true', color='m')
plt.plot(x, mc[:, 0], label='mc', color='c')
plt.plot(x, ub[:, 0], label='ub', color='b')
plt.plot(x, ds[:, 0], label='ds', color='r')
plt.plot(x, bff[:, 0], label='bff', color='g')
plt.title('Q, action 1')
plt.legend()

# Graph Q(s, 1) vs. s.
plt.subplot(1,2,2)
plt.plot(x, true[:, 1], label='true', color='m')
plt.plot(x, mc[:, 1], label='mc', color='c')
plt.plot(x, ub[:, 1], label='ub', color='b')
plt.plot(x, ds[:, 1], label='ds', color='r')
plt.plot(x, bff[:, 1], label='bff', color='g')
plt.title('Q, action 2')
plt.legend()