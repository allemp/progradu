#%%
# Importing libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from load_asr_data import load_keywords, load_transcripts, load_test_transcript
import itertools as it
from nltk.stem.snowball import SnowballStemmer
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri


#%%
# Generator with a rolling window of size 2
def rolling_window(list):
    degree = 2
    for i in range(len(list)-degree+1):
        yield [list[i+o] for o in range(degree)]

#%%
# Create weighted adjacency matrix for keywords
def compute_adjacency_matrix(transcript, keyword_categories):
    gen = rolling_window(transcript)
    edges = []
    concepts = sorted(list(set(keyword_categories.values())))
    keywords = list(set(keyword_categories.keys()))
    adj_matrix = pd.DataFrame(np.zeros(shape=(len(concepts),len(concepts))),columns=concepts, index = concepts)
    for segment in transcript:
        found_keywords = [word for word in segment if word in keywords]
        if len(found_keywords) > 1:
            edges = edges + list(it.combinations(found_keywords,2))

    for window in gen:
        edges = edges + list(it.product([word for word in window[0] if word in keywords], [word for word in window[1] if word in keywords]))
    
    for edge in edges:
        adj_matrix.loc[keyword_categories[edge[0]],keyword_categories[edge[1]]] += 1
        adj_matrix.loc[keyword_categories[edge[1]],keyword_categories[edge[0]]] += 1
    return adj_matrix

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
# Load transcripts and keywords
stemmer = SnowballStemmer("finnish")
transcripts = load_transcripts()


phys_concepts = "sähkövirta,teho,sähköenergia,jännite,paine,energiamuoto,tasavirta,vaihtovirta,valo,voima,säteily,nollapiste,ääni,aalto,energia,resistanssi,taajuus,työ,vaihtojännite,nostotyö,noste,lämpöenergia,lämpötila,massa,taso,kitka,induktio,magneettinen,magneettikenttä,potentiaalienergia,ydinenergia,aine,melu".split(",")
phys_concepts.append("virta")
phys_concepts = [stemmer.stem(word.strip()) for word in phys_concepts]
phys_concepts = dict(zip(phys_concepts,["phys_concepts"]*len(phys_concepts)))

phys_objects = "jännitemittari,akku,automaattisulake,sulake,hehkulamppu,vesihöyry,rautasydän,muuntaja,paristo,napa,ilmaston,tähti,käämi,vastus,halo,tuuli,komponentti,termostaatti,sähkömagneetti,magneetti,akseli,sähköverkko,sarjaan,aurinkopaneeli,oikosulku,loisteputki,sähköisku,lamppu,maalämpö,generaattori,energialamppu,sähkömoottori,maapallo,led,pistotulppa,yleismittari,diodi,fyysikko,vesi,rinnan,johde,aurinko,halogeenilamppu,sähkölaite,verkkojännite,virtamittari,prisma,virtapiiri,kytkentäkaavio".split(",")
phys_objects = [stemmer.stem(word.strip()) for word in phys_objects]
phys_objects = dict(zip(phys_objects,["phys_objects"]*len(phys_objects)))


phys_units = "joule,watti,yksikkö,lukuarvo,ampeeri".split(",")  
phys_units = [stemmer.stem(word.strip()) for word in phys_units]
phys_units = dict(zip(phys_units,["phys_units"]*len(phys_units)))

phys_meta = "teoria,malli,hypoteesi,absoluuttinen".split(",")
phys_meta = [stemmer.stem(word.strip()) for word in phys_meta] 
phys_meta = dict(zip(phys_meta,["phys_meta"]*len(phys_meta)))

keyword_categories = {**phys_concepts,**phys_objects,**phys_units,**phys_meta}

#%%
# Compute adjacency matrices
dataframes = []
for transcript in transcripts:
    dataframes.append(compute_adjacency_matrix(stem_transcript(transcript), keyword_categories))

#%%
dataframes[2]

#%%
