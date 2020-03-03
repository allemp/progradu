#%% Import libraries
import pandas as pd
import numpy as np
import json
from matrixprofile import *

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
    return matrixProfile.stomp(np.asarray(keyword_freq), m)

#%% Compute an adjusted matrix profile
def matrix_profile_adjusted(mp, m):
    return np.append(mp[0], np.zeros(m-1)+np.nan)

#%% 
if __name__ == "__main__":
    keywords_path = "../../data/interim/keywords_voikko.json"
    transcripts_path = "../../data/interim/transcripts_voikko.json"

    with open(keywords_path, "r") as f:
        keywords = json.load(f)
    with open(transcripts_path, "r") as f:
        transcripts = json.load(f)

# %%


# %%
