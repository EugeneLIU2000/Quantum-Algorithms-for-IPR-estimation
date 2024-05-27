import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


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


 
m = -1*np.linspace(-1, 0., 20)
IPR_ED = np.loadtxt('PXP_ED.csv', delimiter=',')
 
IPR_3 = np.loadtxt('PXP_est3.csv', delimiter=',')
IPR_4 = np.loadtxt('PXP_est4.csv', delimiter=',')
IPR_5 = np.loadtxt('PXP_est5.csv', delimiter=',')
IPR_violation = np.loadtxt('PXP_violation.csv', delimiter=',')

 
fig, ax1 = plt.subplots(1,1, figsize = (12, 10))

ax2 = ax1.twinx()

 
ax1.plot(m, IPR_3, ls="dotted", color='blue', linewidth=5, label='$m = 3$')
ax1.plot(m, IPR_4, ls="dashed", color='green', linewidth=5, label='$m = 4$')
ax1.plot(m, IPR_5, ls="dashdot", color='magenta', linewidth=5, label='$m = 5$')
ax1.plot(m, IPR_ED, color='black', linewidth=5, label='ED')
ax2.plot(m, IPR_violation,  ls=(5,(10,3)), color='red', linewidth = 5)

 
ax1.legend(fontsize=24)
 
ax1.set_ylabel(r"$I^Z_2$",fontsize=BIGGER_SIZE, labelpad = 0)
ax2.set_ylabel(r"$\delta \sigma^z$",fontsize=BIGGER_SIZE, labelpad = 20)
ax1.set_xlabel(r"transverse field amplitude $h$",fontsize=BIGGER_SIZE)

ax2.yaxis.label.set_color('red')
ax2.spines['right'].set_color('red')
ax2.yaxis.label.set_color('red')
ax2.tick_params(axis='y', colors='red')
ax2.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5], ["0.1", "0.2", "0.3", "0.4", "0.5"])

ax1.set_yticks([0.22, 0.26, 0.3, 0.34, 0.38], ["0.22", "0.26", "0.3", "0.34", "0.38"])
ax1.axvline(x=0.655, ls = 'dashed', color = 'black', linewidth = 2)


plt.savefig('./fig5.pdf', format='pdf', dpi=400, bbox_inches='tight')


#%%