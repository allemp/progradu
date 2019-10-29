#%% Imports

import os
import sys
import json
import numpy as np
import libvoikko
from nltk.stem.snowball import SnowballStemmer

#%% Create root word using Snowball stemmer

def nltk_stemmer(words):
    stemmer = SnowballStemmer("finnish")
    return [stemmer.stem(word.strip()) for word in words]

#%% Create root word using Voikko lemmatizer

def voikko_lemmatizer(words):
    lemmatizer = libvoikko.Voikko("fi")
    root_words = []

    for word in words:
        v_dict = lemmatizer.analyze(word.strip())
        try: 
            root_words.append(v_dict[0]["BASEFORM"])
        except:
            pass
    
    return root_words


#%% Create a stemmed/lemmatized transcript from a raw text file

def create_transcript(data_path, word_rooter):
    transcript = []
    try:
        with open(data_path, encoding = "UTF-8") as fp:
            lines = fp.readlines()
        for line in lines:
            segment = line.split(" ")[2:]
            transcript.append(word_rooter(segment))
    except:
        print("File " + data_path + " was not found")
    return transcript

#%% Create a stemmed/lemmatized keyword list from a raw text file

def create_keywordlist(data_path, word_rooter):
    with open(data_path, encoding = "UTF-8") as fp:
        keywords = word_rooter(fp.readlines())
    return keywords


#%% Create dataset 

def create_dataset():
    keywords_path = "data/raw/keywords/master_keywords_unstemmed.txt"
    transcripts_path = "data/raw/transcripts/"

    keywords_stemmed = create_keywordlist(keywords_path , nltk_stemmer)
    keywords_voikko = create_keywordlist(keywords_path , voikko_lemmatizer)

    transcripts_stemmed = []
    transcripts_voikko = []

    for file in os.listdir(transcripts_path):
        transcripts_stemmed.append(create_transcript(os.path.join(transcripts_path, file) , nltk_stemmer))
        transcripts_voikko.append(create_transcript(os.path.join(transcripts_path, file) , voikko_lemmatizer))

    return (keywords_stemmed, transcripts_stemmed, keywords_voikko, transcripts_voikko)


#%% Write dataset to disk if run directly

if __name__ == "__main__":

    keywords_stemmed, transcripts_stemmed, keywords_voikko, transcripts_voikko = create_dataset()

    with open('data/interim/keywords_stemmed.json', 'w') as f:
        json.dump(keywords_stemmed, f, sort_keys=True)

    with open('data/interim/keywords_voikko.json', 'w') as f:
        json.dump(keywords_voikko, f, sort_keys=True)

    with open('data/interim/transcripts_stemmed.json', 'w') as f:
        json.dump(transcripts_stemmed, f, sort_keys=True)

    with open('data/interim/transcripts_voikko.json', 'w') as f:
        json.dump(transcripts_voikko, f, sort_keys=True)

#%%
