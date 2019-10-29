#%% Importing libraries
import pandas as pd
import numpy as np
import json

#%% Load transcripts and keywords
stemmer = SnowballStemmer("finnish")
transcripts = load_transcripts()

phys_concepts = "sähkövirta,teho,sähköenergia,jännite,paine,energiamuoto,tasavirta,vaihtovirta,valo,voima,säteily,nollapiste,ääni,aalto,energia,resistanssi,taajuus,työ,vaihtojännite,nostotyö,noste,lämpöenergia,lämpötila,massa,taso,kitka,induktio,magneettinen,magneettikenttä,potentiaalienergia,ydinenergia,aine,melu".split(",")
phys_objects = "jännitemittari,akku,automaattisulake,sulake,hehkulamppu,vesihöyry,rautasydän,muuntaja,paristo,napa,ilmaston,tähti,käämi,vastus,halo,tuuli,komponentti,termostaatti,sähkömagneetti,magneetti,akseli,sähköverkko,sarjaan,aurinkopaneeli,oikosulku,loisteputki,sähköisku,lamppu,maalämpö,generaattori,energialamppu,sähkömoottori,maapallo,led,pistotulppa,yleismittari,diodi,fyysikko,vesi,rinnan,johde,aurinko,halogeenilamppu,sähkölaite,verkkojännite,virtamittari,prisma,virtapiiri,kytkentäkaavio".split(",")
phys_units = "joule,watti,yksikkö,lukuarvo,ampeeri".split(",")  
phys_meta = "teoria,malli,hypoteesi,absoluuttinen".split(",")

#%% Stem the words in the transcript using Snowball stemmer
def stem_transcript(transcript):
    stemmer = SnowballStemmer("finnish")
    segments = []
    for segment in transcript:
        words = []
        for word in segment["words"]:
            words.append(stemmer.stem(word.strip()))
        segments.append(words)
    return segments

#%% Stem transcripts and keywords

phys_concepts = [stemmer.stem(word.strip()) for word in phys_concepts]
phys_objects = [stemmer.stem(word.strip()) for word in phys_objects]
phys_units = [stemmer.stem(word.strip()) for word in phys_units]
phys_meta = [stemmer.stem(word.strip()) for word in phys_meta] 

transcripts = [stem_transcript(transcript) for transcript in transcripts]

keywords = {"concepts": phys_concepts, "objects": phys_objects, "units": phys_units, "meta": phys_meta}
#%% Concatenate 5 second segments into a window
def transcript_window(transcript, size):
    new_transcript = []
    for i in range(0,len(transcript),size):
        # Flatten list
        new_transcript.append(sum(transcript[i:i+size], []))
    return new_transcript
#%%
# 60 segments is 5min
size = 60
new_transcripts = [transcript_window(transcript,size) for transcript in transcripts]

#%% Count keywords in given segment
def keyword_counter(window,keywords):
    result = {}
    for key, value in keywords.items():
        count = 0
        for word in value:
            count += window.count(word)
        result[key] = count
    result["other"] = len(window) - sum(result.values())
    return result

#%% Example of stemmer issue, sähkövir and sähkövirt, try Voikko
keyword_counter(new_transcripts[0][2][100:120],keywords)

#%% Wrapper to count keywords in a given transcript
def transcript_keyword_counter(transcript, keywords):
    result = []
    for segment in transcript:
        result.append(keyword_counter(segment,keywords))
    return result
#%%
teacher = 5

transcript_keyword_counter(new_transcripts[0],keywords)

#%%
pd.DataFrame(transcript_keyword_counter(new_transcripts[teacher],keywords)).plot(subplots = True, layout = (5,1))
print("Learning gain " + str(pd.read_csv("data/learning_gain.csv")["learning_gain"][teacher]))


#%%
teacher_data = []
for t in new_transcripts:
    teacher_data.append(pd.DataFrame(transcript_keyword_counter(t,keywords))["concepts"])
pd.concat(teacher_data, axis = 1).plot(subplots = True, layout = (25,1), figsize=(10,25),  ylim = (0,20))
#%%
