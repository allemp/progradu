#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from physicsasr.dataset.create_dataset import Dataset
from physicsasr.features.create_features import Features

#%%
dataset = Dataset("/workspaces/speechrecognition/data/interim/keywords_voikko.json", "/workspaces/speechrecognition/data/interim/transcripts_voikko.json")
features = Features(dataset.transcripts[2], dataset.keywords)

keyword_window = 2

m = 120

counts = features.compute_keyword_freq(keyword_window)
mp_adj = features.compute_transcript_mp(keyword_window,m)
cac = features.compute_cac(keyword_window, m, m,2)

#%%
fig, (ax1, ax2, ax3) = plt.subplots(3,1)

ax1.plot(np.arange(len(counts)), counts, label="Keyword frequency")
ax1.set_ylabel("Keyword count")

ax2.plot(np.arange(len(mp_adj)),mp_adj, label="Matrix Profile", color = "red")
ax2.set_ylabel('Matrix Profile')

ax3.plot(np.arange(len(cac)), label="Corrected Arc Curve", color="green")
ax3.set_ylabel("CAC")


# %%
