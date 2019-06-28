#%%
# Importing libraries
import pandas as pd
import numpy as np
from load_asr_data import load_keywords, load_transcripts

from nltk.stem.snowball import SnowballStemmer
import nltk

#%% Load transcripts and keywords
transcripts = load_transcripts()
keywords = load_keywords()
keywords.append("virta")

stemmer = SnowballStemmer("finnish")
keywords_stemmed = [stemmer.stem(word.strip()) for word in keywords]

words_stemmed = []
for segment in transcripts[19]:
    for word in segment["words"]:
        words_stemmed.append(stemmer.stem(word.strip()))

#%%
found_keywords = [word for word in words_stemmed if word in keywords_stemmed]
nltk.FreqDist(found_keywords)