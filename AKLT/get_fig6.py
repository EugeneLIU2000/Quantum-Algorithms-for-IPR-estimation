import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

 


N_sites = 4
filename_IPR_QC = "IPR_AKLT_algorithm_N_AKLT_sites." + str(N_sites) + ".dat"
filename_IPR_ED = "IPR_AKLT_ED_N_AKLT_sites." + str(N_sites) + ".dat"

data_IPR_QC = np.loadtxt(filename_IPR_QC)
data_IPR_ED = np.loadtxt(filename_IPR_ED)


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

 
ax.plot(data_IPR_ED[:,0], data_IPR_ED[:,1], color="black", linewidth=4,  label="ED")

ax.plot(data_IPR_QC[:,0], data_IPR_QC[:,1], color='red', linestyle='dashed',linewidth = 4, label="simulation")
  
ax.legend(fontsize=24)

ax.set_xlabel(r"transverse field amplitude h", fontsize=BIGGER_SIZE, labelpad = 0)
ax.set_ylabel(r"$I^Z_2$", fontsize=BIGGER_SIZE, labelpad = 0)

ax.set_yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1"])

 
plt.savefig('./fig6.pdf', format='pdf', dpi=400, bbox_inches='tight')

 