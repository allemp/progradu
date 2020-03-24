#%% Import libraries
import pandas as pd
import numpy as np
import json
#from matrixprofile import *
import stumpy

#%% Counting occurances of a keyword in a list of words
def keyword_counter(words, keywords):
    count = 0
    keywords = set(keywords)
    for word in words:
        if word in keywords:
            count += 1
    return count


#%% Fixed window function
def fixed_window(seq, size):
    for i in range(0, len(seq),size):
        yield sum(seq[i:i+size],[])

#%% Count the keyword frequency for a transcript
def transcript_keyword_freq(transcript, keywords, window_size):
    return [keyword_counter(words, keywords) for words in list(fixed_window(transcript,window_size))]

#%% Compute a matrix profile
def matrix_profile(keyword_freq, m):
    return stumpy.stump(np.asarray(keyword_freq, dtype=np.double), m=m)


#%% Compute an adjusted matrix profile from a matrix profile
def matrix_profile_adjusted(mp, m):
    return np.append(mp[:,0], np.zeros(m-1)+np.nan)

#%% Compute a corrected arc curve (CAC) using Fluss algorithm from a matrix profile
def cac_fluss(mp, L, n_regimes):
    return stumpy.fluss(mp[: , 1], L = L, n_regimes = n_regimes, excl_factor = 1)

#%% Features class to obtain the keyword frequencies and matrix profiles
class Features:

    def __init__(self, transcript, keywords):
        self.transcript = transcript
        self.keywords = keywords

    def compute_transcript_mp(self, window_size, mp_window):
        return matrix_profile_adjusted(matrix_profile(
            transcript_keyword_freq(self.transcript,self.keywords,window_size),
            mp_window),mp_window)

    def compute_keyword_freq(self, window_size):
        return transcript_keyword_freq(self.transcript, self.keywords, window_size)

    def compute_cac(self, window_size, mp_window, L, n_regimes):
        return cac_fluss(matrix_profile(
            transcript_keyword_freq(self.transcript,self.keywords,window_size),
            mp_window), L, n_regimes)