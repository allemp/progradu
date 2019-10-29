#%%
# Importing libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from load_asr_data import load_keywords, load_transcripts

import itertools as it
from nltk.stem.snowball import SnowballStemmer

#%% Load transcripts and keywords
transcripts = load_transcripts()
keywords = load_keywords()

keywords.append("virta")


stemmer = SnowballStemmer("finnish")
keywords_stemmed = [stemmer.stem(word.strip()) for word in keywords]

#%%
# Rolling window helper function and creating a generator
def rolling_window(list, degree):
    for i in range(len(list)-degree+1):
        yield [list[i+o] for o in range(degree)]
gen = rolling_window(transcripts[20], 2)

#%%
# Making list of connected keywords
concept_connections = []
for window in gen:
    connected_concepts = set()
    for word in window[0]["words"]:
        if stemmer.stem(word.strip()) in keywords_stemmed:
            connected_concepts.add(stemmer.stem(word.strip()))
    for word in window[1]["words"]:
        if stemmer.stem(word.strip()) in keywords_stemmed:
            connected_concepts.add(stemmer.stem(word.strip()))
    if len(connected_concepts) > 1:
        concept_connections.append(list(it.combinations(connected_concepts,2)))


#%%
# Making graph
G = nx.Graph()
for connections in concept_connections:
    G.add_edges_from(connections)
nx.draw(G, with_labels = True)
plt.draw()

#%%
# Compute degree centrality
print(pd.DataFrame.from_dict(nx.degree_centrality(G), orient = "index").sort_values(0, ascending = False))
#%%
# Compute closeness centrality
print(pd.DataFrame.from_dict(nx.closeness_centrality(G), orient = "index").sort_values(0, ascending = False))
#%%
# Compute betweenness centrality
print(pd.DataFrame.from_dict(nx.betweenness_centrality(G), orient = "index").sort_values(0, ascending = False))
#%%
# Compute harmonic centrality
print(pd.DataFrame.from_dict(nx.harmonic_centrality(G), orient = "index").sort_values(0, ascending = False))

#%%
# Compute pagerank
print(pd.DataFrame.from_dict(nx.pagerank(G), orient = "index").sort_values(0, ascending = False))