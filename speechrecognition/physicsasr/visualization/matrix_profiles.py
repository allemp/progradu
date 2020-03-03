import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


fig, (ax1, ax2) = plt.subplots(2,1)

ax1.plot(np.arange(len(counts)), counts, label="Keyword frequency")
ax1.set_ylabel("Keyword count")

ax2.plot(np.arange(len(mp_adj)),mp_adj, label="Matrix Profile", color = "red")
ax2.set_ylabel('Matrix Profile')

#ax3.plot(np.arange(len(cac)), label="Corrected Arc Curve", color="green")
#ax3.set_ylabel("CAC")
