#%% 
# Importing libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import time
from nltk.stem.snowball import SnowballStemmer
from nltk.metrics import edit_distance, jaccard_distance
from nltk.metrics.distance import jaro_similarity, jaro_winkler_similarity
from nltk.corpus import stopwords 

from load_asr_data import load_keywords, load_transcripts
#%%
# Loading data and creating stemmer
transcripts = load_transcripts()
keywords = load_keywords()
stop_words = stopwords.words('finnish')

stemmer = SnowballStemmer("finnish")

#%%
# Stem stopwords, keywords, words
stop_words_stemmed = set([stemmer.stem(word.strip()) for word in stop_words])
keywords_stemmed = [stemmer.stem(word.strip()) for word in keywords]
words_stemmed = []
words = []
for transcript in transcripts:
    for segment in transcript:
        for word in segment["words"]:
            words_stemmed.append(stemmer.stem(word.strip()))
            words.append(word)
#%%
# Load lemmatized stopwords, keywords, words
# cat data/master_keywords_unstemmed.txt | docker run -i turkunlp/turku-neural-parser:finnish-cpu-plaintext-stdin > keywords_lemma.tsv
# sed '/^#/ d' < keywords_lemma.tsv > keywords_lemma2.tsv

keywords_lemmatized = pd.read_csv("keywords_lemma2.tsv", sep = "\t", header = None)[2].str.replace("#","").tolist()

with open('stop_words.txt', 'w', encoding = "UTF-8") as f:
    for item in stop_words:
        f.write("%s\n" % item)
#%%
#  cat stop_words.txt | docker run -i turkunlp/turku-neural-parser:finnish-cpu-plaintext-stdin > stopwords_lemma.tsv
# sed '/^#/ d' < stopwords_lemma.tsv > stopwords_lemma2.tsv
stop_words_lemmatized = pd.read_csv("stopwords_lemma2.tsv", sep = "\t", header = None)[2].str.replace("#","").tolist()

with open('words.txt', 'w', encoding = "UTF-8") as f:
    for item in words:
        f.write("%s\n" % item)

#%%
# cat words.txt | docker run -i turkunlp/turku-neural-parser:finnish-cpu-plaintext-stdin > words_lemma.tsv
# sed '/^#/ d' < words_lemma.tsv > words_lemma2.tsv
words_lemmatized = pd.read_csv("words_lemma2.tsv", sep = "\t", header = None)[2].str.replace("#","").tolist()


#%%
# Remove stopwords
#keywords_preprocessed = keywords_stemmed
keywords_preprocessed = keywords_lemmatized
#words_preprocessed = [w for w in words_stemmed if not w in stop_words_stemmed]
words_preprocessed = [w for w in words_lemmatized if not w in stop_words_lemmatized]
words_preprocessed = words_preprocessed[0:1000]

#%%
# Calculate the edit distance for each word and keyword
def get_dists(keyword):
    dists = []
    for word in words_preprocessed:
        dists.append({"edit_dist": edit_distance(word,keyword), "jaro_simi": jaro_similarity(word,keyword), "jaro_winkler_simi": jaro_winkler_similarity(word,keyword) , "jaccard_dist": jaccard_distance(set(word),set(keyword)), "word": word, "keyword": keyword})
    return pd.DataFrame(dists).sort_values("edit_dist").iloc[0:3,:]

edit_distances = []
t0 = time.time()
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(get_dists)(keyword) for keyword in keywords_preprocessed)
t1 = time.time()

print (t1-t0)
#%%
distances_df = pd.concat(results)
#%%
#distances_df[distances_df["edit_dist"] < 5].sort_values("edit_dist").to_excel("distances_stem.xlsx")
#distances_df[distances_df["edit_dist"] < 5].sort_values("edit_dist").to_excel("distances_lemm.xlsx")
#%%
distances_df
#%%
