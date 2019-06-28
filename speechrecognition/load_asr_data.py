#%%
# importing libraries
#%%
# Load data from T1-T27 transcription files
def load_transcripts():
    data = []
    for i in range(1,28):
        path = "data/T" + str(i) + ".txt"
        try:
            lines = []
            with open(path, encoding = "UTF-8") as fp:
                lines = fp.readlines()
            text = []
            for line in lines:
                line_list = line.split(" ")
                line_dict = {"start": line_list[0], "end": line_list[1], "words": line_list[2:]}
                text.append(line_dict)
            data.append(text)
        except:
            print("File " + path + " failed")
    return data
#%%
# Load unstemmed master keyword list
def load_keywords():
    master_keywords_unstemmed = []
    keywords_path = "data/master_keywords_unstemmed.txt"
    try:
        with open(keywords_path, encoding = "UTF-8") as fp:
            master_keywords_unstemmed = fp.readlines()
    except:
        print("File " + keywords_path + " failed")
    return master_keywords_unstemmed

#%%
#%%
# Load test data
def load_test_transcript():
    data = []
    path = "data/test_data" + ".txt"
    try:
        lines = []
        with open(path, encoding = "UTF-8") as fp:
            lines = fp.readlines()
        text = []
        for line in lines:
            line_list = line.split(" ")
            line_dict = {"start": line_list[0], "end": line_list[1], "words": line_list[2:]}
            text.append(line_dict)
        data.append(text)
    except:
        print("File " + path + " failed")
    return data
#%%
# Load test keywords
def load_test_keywords():
    keywords = []
    keywords_path = "data/test_keywords.txt"
    try:
        with open(keywords_path, encoding = "UTF-8") as fp:
            keywords = fp.readlines()
    except:
        print("File " + keywords_path + " failed")
    return keywords