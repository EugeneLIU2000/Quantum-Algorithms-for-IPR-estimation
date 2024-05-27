import matplotlib.pyplot as plt
import numpy as np

time = np.linspace(0.0, 0.5*3.14, 50)
time_n = time[0:50:2]

IPR_ED = np.loadtxt('IPR_ED.csv', delimiter=',')
torino_trial1 = np.loadtxt('torino_trial1.csv', delimiter=',')
torino_trial2 = np.loadtxt('torino_trial2.csv', delimiter=',')
torino_trial3 = np.loadtxt('torino_trial3.csv', delimiter=',')
torino_trial4 = np.loadtxt('torino_trial4.csv', delimiter=',')
torino_trial5 = np.loadtxt('torino_trial5.csv', delimiter=',')
simulation = np.loadtxt('IPR_algorithm.csv', delimiter=',')


IPR_1 = []
IPR_2 = []
IPR_3 = []
IPR_4 = []
IPR_5 = []

# From (quasi-)probability to IPR
for i in range(25):
    IPR_1.append(2*(torino_trial1[i] - 0.5))
    IPR_2.append(2*(torino_trial2[i] - 0.5))
    IPR_3.append(2*(torino_trial3[i] - 0.5))
    IPR_4.append(2*(torino_trial4[i] - 0.5))
    IPR_5.append(2*(torino_trial5[i] - 0.5))


std = []
mean = []

for i in range(25):
    datalist = [IPR_1[i], IPR_2[i], IPR_3[i], IPR_4[i], IPR_5[i]]
    std.append(np.std(datalist))
    mean.append(np.mean(datalist))

print("std", std)
# print(std)

# This is an individual data run for t=0, the transpilation optimizing the circuit into a nearly idle circuit
t0_result = [0.0242, 0.0031, 0.0148, 0.0056, 0.0321, 0.0291, 0.0038, 0.006, 0.0122, 0.0811]
std[0] = np.std(t0_result)
mean[0] = 1 - np.mean(t0_result)

#%%

SMALL_SIZE = 26
MEDIUM_SIZE = 26
BIGGER_SIZE = 30

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
 
fig, ax = plt.subplots(1,1, figsize = (12, 10))

x = time/np.pi
ax.plot(x, IPR_ED, color="black", linewidth=4,  label="ED")

ax.plot(x, simulation, color='red', linestyle='dashed',linewidth = 4, label="simulation")

y_IBM = np.array(mean)
dy_IBM = np.array(std)
x_n_IBM = time_n/np.pi

idx_to_remove = 11
y_IBM = np.delete(y_IBM, idx_to_remove)
x_n_IBM = np.delete(x_n_IBM, idx_to_remove)
dy_IBM = np.delete(dy_IBM, idx_to_remove)

idx_to_remove = 13
y_IBM = np.delete(y_IBM, idx_to_remove)
x_n_IBM = np.delete(x_n_IBM, idx_to_remove)
dy_IBM = np.delete(dy_IBM, idx_to_remove)

idx_to_remove = 21
y_IBM = np.delete(y_IBM, idx_to_remove)
x_n_IBM = np.delete(x_n_IBM, idx_to_remove)
dy_IBM = np.delete(dy_IBM, idx_to_remove)

ax.errorbar(x_n_IBM, y_IBM, dy_IBM,  marker=".", linestyle='None',  markersize=24, color="blue", ecolor="blue", capsize=3, elinewidth=0.5, label=r"experiment")
 
ax.legend(fontsize=24)

ax.set_xlabel(r"time $t/\pi$", fontsize=BIGGER_SIZE, labelpad = 0)
ax.set_ylabel(r"$I^X_2$", fontsize=BIGGER_SIZE, labelpad = 0)

ax.set_yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1"])

ax.axvline(x=0.25, ls = 'dashed', color = 'black', linewidth = 2)

plt.savefig('./fig7.pdf', format='pdf', dpi=400, bbox_inches='tight')


data_IBM_experiment = np.zeros((y_IBM.shape[0], 3))
data_IBM_experiment[:,0] = x_n_IBM
data_IBM_experiment[:, 1] = y_IBM
data_IBM_experiment[:, 2] = dy_IBM

np.savetxt("./data_IBM_experiment.txt", data_IBM_experiment)



