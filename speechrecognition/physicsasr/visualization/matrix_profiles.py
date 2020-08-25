#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from physicsasr.dataset.create_dataset import Dataset
from physicsasr.features.create_features import Features
from matrixprofile import *

#%%
dataset = Dataset("/workspaces/speechrecognition/data/interim/keywords_voikko.json", "/workspaces/speechrecognition/data/interim/transcripts_voikko.json")

keyword_window = 12
m = 15
# Single lesson
transcripts = dataset.transcripts

#All lessons appended
lesson_lengths = [len(transcript)/keyword_window for transcript in dataset.transcripts]

transcripts = [[segment for transcript in dataset.transcripts for segment in transcript]]

# Comparing two lessons

keywords = dataset.keywords
#%%
for transcript in dataset.transcripts:
    features = Features(transcript["transcript"], keywords, transcript["teacher"], transcript["lesson"])
    freqs = features.compute_keyword_freq(keyword_window)
    
    mp = features.compute_transcript_mp(keyword_window,m)
    mp_adj = np.append(mp[0], np.zeros(m-1)+np.nan)
    #cac = features.compute_cac(mp, m)

    fig, (ax1, ax2) = plt.subplots(2,1, sharex = 'col')

    ax1.plot(np.arange(len(freqs["freq_keywords"])), freqs["freq_keywords"], label="Keyword frequency")
    ax1.set_ylabel("Keyword count")
    ax1.set_ylim(ymin=0)

    ax2.plot(np.arange(len(mp_adj)),mp_adj, label="Matrix Profile", color = "red")
    ax2.set_ylabel('Matrix Profile')
    ax2.set_ylim(ymin=0)

    motifs_i, motifs_d = motifs.motifs(np.asarray(freqs["freq_keywords"], dtype=np.double), mp, max_motifs = 10)

    #motif1 = motifs_i[2][0]
    #motif2 = motifs_i[2][1]


    #print(freqs["freq_keywords"][(motif1-m):(motif1+m)])
    #print(transcript[((motif1 * keyword_window) - keyword_window + 1): (motif1 * keyword_window)])

    #print(freqs["freq_keywords"][(motif2-m):(motif2+m)])
    #print(transcript[((motif2 * keyword_window) - keyword_window + 1): (motif2 * keyword_window)])