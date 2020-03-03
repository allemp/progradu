#%% Imports

import os
import sys
import argparse
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

#%% Read a file from disk from the specified path

def read_file(data_path):
    lines = []
    try:
        with open(data_path, encoding = "UTF-8") as fp:
            lines = fp.readlines()
    except:
        print("File " + data_path + " was not found")
    return lines

#%% Create a stemmed/lemmatized transcript from a raw text file

def create_transcript(lines, baseformer):
    transcript = []
    for line in lines:
        segment = line.split(" ")[2:]
        transcript.append(baseformer(segment))
    
    return transcript

#%% Create a stemmed/lemmatized keyword list from a raw text file

def create_keywordlist(lines, baseformer):
    keywords = baseformer(lines)
    return keywords

#%% Write interim data to stdout when run from the command line

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required = True)
    parser.add_argument("--type", required = True, choices = ["keywords","transcripts"])
    parser.add_argument("--baseformer", required = True, choices = ["voikko", "nltk"])

    args = parser.parse_args()

    baseformers = {"voikko": voikko_lemmatizer, "nltk": nltk_stemmer}

    if args.type == "keywords":
        print(json.dumps(create_keywordlist(read_file(args.path), baseformers[args.baseformer])))
    
    elif args.type == "transcripts":
        transcripts = []
        for file in os.listdir(args.path):
            transcripts.append(create_transcript(read_file(os.path.join(args.path, file)) , baseformers[args.baseformer]))
        print(json.dumps(transcripts))

    else:
        raise NotImplementedError