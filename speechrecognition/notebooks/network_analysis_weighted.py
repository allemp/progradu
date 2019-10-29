#%%
# Importing libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from load_asr_data import load_keywords, load_transcripts, load_test_transcript, load_learning_gain
import itertools as it
from nltk.stem.snowball import SnowballStemmer
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from sklearn import linear_model

# Installing tnet for R
# from rpy2.robjects.packages import importr
# utils = importr('utils')
 #utils.install_packages('tnet')


#%%
# Generator with a rolling window of size 2
def rolling_window(list):
    degree = 2
    for i in range(len(list)-degree+1):
        yield [list[i+o] for o in range(degree)]

#%%
# Create weighted adjacency matrix for keywords
def get_edges(transcript, keywords):
    gen = rolling_window(transcript)
    edges = []
    sorted_edges = []
    for segment in transcript:
        found_keywords = [word for word in segment if word in keywords]
        if len(found_keywords) > 1:
            edges = edges + list(it.combinations(found_keywords,2))

    for window in gen:
        edges = edges + list(it.product([word for word in window[0] if word in keywords], [word for word in window[1] if word in keywords]))
    
    for edge in edges:
        sorted_edges.append(tuple(sorted(edge)))
    return sorted_edges

#%% 
# Stem transcripts
def stem_transcript(transcript):
    stemmer = SnowballStemmer("finnish")
    segments = []
    for segment in transcript:
        words = []
        for word in segment["words"]:
            words.append(stemmer.stem(word.strip()))
        segments.append(words)
    return segments

#%%
# Edge to tnet format (both directions) pandas
def edge_to_tnet(edge):
    return [{"from":edge[0], "to":edge[1], "weight":edge[2]},
            {"from":edge[1], "to":edge[0], "weight":edge[2]}]

#%%
# Apply tnet function to dataframe
def tnet_apply_func(df,func,result_index):
    mapping = dict(enumerate(df['from'].cat.categories))
    r_tnet_df = pd.concat([tnet_df["from"].cat.codes + 1,tnet_df["to"].cat.codes + 1, tnet_df["weight"]],axis = 1).astype("int64")

    results = pd.DataFrame(np.array(func(r_tnet_df)))
    return results.apply(lambda x: pd.Series({"keyword":mapping[x[0] - 1],"output":x[result_index]}), axis = 1)

#%%
# Count weights for edge list
def count_edges(edges):
    weighted_edges = []
    for edge in edges:
        weighted_edge = edge + (edges.count(edge),)
        if weighted_edge not in weighted_edges:
            weighted_edges.append(weighted_edge)
    return weighted_edges

#%% 
# Load transcripts and keywords
stemmer = SnowballStemmer("finnish")
transcripts = load_transcripts()
keywords = load_keywords()
keywords.append("virta")
keywords_stemmed = [stemmer.stem(word.strip()) for word in keywords]

#%%
# Get list of occuring keywords
transcript_list = sum(sum([stem_transcript(t) for t in transcripts],[]),[])
keyword_list = keywords_stemmed
occuring_keywords = list(set(transcript_list).intersection(keywords_stemmed))
keywords_degree = [word + "_degree" for word in occuring_keywords]
keywords_betweenness = [word + "_betweenness" for word in occuring_keywords]
keywords_closeness_normalized = [word + "_closeness_normalized" for word in occuring_keywords]

#%%
# Calculate weighted degree, betweenness and closeness for each teacher for each keyword
teacher_measures_list = []
teacher_weighted_edges = []

for transcript in transcripts:
    weighted_edges = count_edges(get_edges(stem_transcript(transcript), keywords_stemmed))
    teacher_weighted_edges.append(weighted_edges)

    tnet_df = pd.DataFrame(sum([edge_to_tnet(edge) for edge in weighted_edges],[]))
    tnet_df["from"] = tnet_df["from"].astype("category")
    tnet_df["to"] = tnet_df["to"].astype("category")

    pandas2ri.activate()
    tnet = importr("tnet")

    teacher_measures = {el:0 for el in keywords_degree + keywords_betweenness + keywords_closeness_normalized}

    degree = tnet_apply_func(tnet_df, tnet.degree_w, 2)
    degree["keyword"] = degree["keyword"] + "_degree"
    for index, row in degree.iterrows():
        teacher_measures[row["keyword"]] = row["output"]


    betweenness = tnet_apply_func(tnet_df, tnet.betweenness_w, 1)
    betweenness["keyword"] = betweenness["keyword"] + "_betweenness"
    for index, row in betweenness.iterrows():
        teacher_measures[row["keyword"]] = row["output"]

    closeness_normalized = tnet_apply_func(tnet_df, tnet.closeness_w, 2)
    closeness_normalized["keyword"] = closeness_normalized["keyword"] + "_closeness_normalized"
    for index, row in closeness_normalized.iterrows():
        teacher_measures[row["keyword"]] = row["output"]
    teacher_measures_list.append(teacher_measures)


#%%
data = pd.DataFrame(teacher_measures_list)
learning_gain = load_learning_gain()

#%%
reg = linear_model.Lasso(alpha = 0.1)
reg = reg.fit(data, learning_gain["learning_gain"])
reg.score(data,learning_gain["learning_gain"])
reg.coef_
#%%
# Create graph

teacher = 20
G = nx.Graph()
G.add_weighted_edges_from(teacher_weighted_edges[teacher])

weights = [w for u,v,w in teacher_weighted_edges[teacher]]

old_min = min(weights)
old_range = max(weights) - old_min
new_min = 1
new_range = 5 + 0.9999999999 - new_min
normalized_weights = [int((n - old_min) / old_range * new_range + new_min) for n in weights]

nx.draw(G, with_labels = True, width = normalized_weights)
plt.draw()

#%%
