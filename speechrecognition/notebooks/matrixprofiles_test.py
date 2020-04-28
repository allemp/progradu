#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stumpy
from physicsasr.dataset.create_dataset import Dataset
from physicsasr.features.create_features import Features

#%%
arrays = np.random.randint(10, size=(3, 1000)).astype("double")

arrays[0,25:30] = np.array([15,20,25,20,15])
arrays[0,65:70] = np.array([14,22,23,22,14])

arrays[1,22:27] = np.array([20,20,20,20,20])
arrays[1,63:68] = np.array([20,20,20,20,20])

arrays[2,22:27] = np.array([15,16,17,18,19])
arrays[2,66:71] = np.array([16,17,18,19,20])
mdmp = stumpy.mstump(arrays, 25)

#%%
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6,1, sharex = 'col')

ax1.plot(np.arange(len(arrays[0,:])), arrays[0,:])
ax1.set_ylabel("A1")
ax1.set_ylim(ymin=0)

ax2.plot(np.arange(len(arrays[1,:])), arrays[1,:])
ax2.set_ylabel("A2")
ax2.set_ylim(ymin=0)

ax3.plot(np.arange(len(arrays[2,:])), arrays[2,:])
ax3.set_ylabel("A3")
ax3.set_ylim(ymin=0)

ax4.plot(np.arange(len(mdmp[0][:,0])),mdmp[0][:,0], color = "red")
ax4.set_ylabel('MP 1D')
ax4.set_ylim(ymin=0)

ax5.plot(np.arange(len(mdmp[0][:,1])),mdmp[0][:,1], color = "red")
ax5.set_ylabel('MP 2D')
ax5.set_ylim(ymin=0)

ax6.plot(np.arange(len(mdmp[0][:,2])),mdmp[0][:,2], color = "red")
ax6.set_ylabel('MP 3D')
ax6.set_ylim(ymin=0)

#%%
print(mdmp[0][:,0].min())
print(mdmp[0][:,1].min())
print(mdmp[0][:,2].min())


# %%
