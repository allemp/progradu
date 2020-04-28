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
#%%
def transcript_keyword_freq(transcript, keywords, window_size):
    phys_objects = "jännitemittari,akku,automaattisulake,sulake,hehkulamppu,vesihöyry,rautasydän,muuntaja,paristo,napa,ilmaston,tähti,käämi,vastus,halo,tuuli,komponentti,termostaatti,sähkömagneetti,magneetti,akseli,sähköverkko,sarjaan,aurinkopaneeli,oikosulku,loisteputki,sähköisku,lamppu,maalämpö,generaattori,energialamppu,sähkömoottori,maapallo,led,pistotulppa,yleismittari,diodi,fyysikko,vesi,rinnan,johde,aurinko,halogeenilamppu,sähkölaite,verkkojännite,virtamittari,prisma,virtapiiri,kytkentäkaavio".split(",")

    phys_concepts = "sähkövirta,teho,sähköenergia,jännite,paine,energiamuoto,tasavirta,vaihtovirta,valo,voima,säteily,nollapiste,ääni,aalto,energia,resistanssi,taajuus,työ,vaihtojännite,nostotyö,noste,lämpöenergia,lämpötila,massa,taso,kitka,induktio,magneettinen,magneettikenttä,potentiaalienergia,ydinenergia,aine,melu".split(",")

    freq_objects = []
    freq_concepts = []
    freq_other = []
    freq_keywords = []
    for words in list(fixed_window(transcript,window_size)):
        objects = keyword_counter(words, phys_objects)
        concepts = keyword_counter(words, phys_concepts)
        other = len(words) - objects - concepts
        keywords1 = keyword_counter(words, keywords)

        freq_objects.append(objects)
        freq_concepts.append(concepts)
        freq_other.append(other)
        freq_keywords.append(keywords1)



    return {"freq_objects": freq_objects, "freq_concepts": freq_concepts, "freq_other":freq_other, "freq_keywords": freq_keywords}

#%% Compute a matrix profile
def matrix_profile(keyword_freq, m):
    return matrixProfile.stomp(np.asarray(keyword_freq["freq_keywords"], dtype=np.double), m)

#%% Compute a corrected arc curve (CAC) using Fluss algorithm from a matrix profile
def cac_fluss(mp, m):
    return fluss.fluss(mp[1], m = m)

#%% Features class to obtain the keyword frequencies and matrix profiles
class Features:

    def __init__(self, transcript, keywords):
        self.transcript = transcript
        self.keywords = keywords

    def compute_transcript_mp(self, window_size, mp_window):
        return matrix_profile(
            transcript_keyword_freq(self.transcript,self.keywords,window_size), mp_window)

    def compute_keyword_freq(self, window_size):
        return transcript_keyword_freq(self.transcript, self.keywords, window_size)

    def compute_cac(self, mp, m):
        return cac_fluss(mp, m)