#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from physicsasr.dataset.create_dataset import Dataset
from physicsasr.features.create_features import Features

import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
#%% 

def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]

# %%
dataset = Dataset("/workspaces/speechrecognition/data/interim/keywords_voikko.json", "/workspaces/speechrecognition/data/interim/transcripts_voikko.json")

counts = []
for transcript in dataset.transcripts:
    chunk_list = []
    i = 0
    for chunk in list(chunks(transcript["transcript"],3)):
        i = i + 1
        features = Features(chunk, dataset.keywords, transcript["teacher"], transcript["lesson"])
        counts2 = features.keyword_category_count()
        counts2["chunk"] = i
        chunk_list.append(counts2)
    counts.append(pd.concat(chunk_list, axis = 0))


data = pd.concat(counts, axis = 0, ignore_index= True)

data = data.set_index("teacher")
raw = data.copy()
data = data.drop("lesson", 1)
data = data.drop("chunk", 1)

def row_minmax(row):
    return (row)/(pd.Series.max(row))



#%%
data = data.drop("meta", 1)
#data = data.drop("other", 1)
data = data.drop("units", 1)


data = pd.DataFrame.apply(data,row_minmax,axis = 1)

#%%
X = data.copy()
y = data.index.values#raw["chunk"]



#%%
linkage = "average"
distance = "cityblock"
k = 3

plt.figure(figsize=(10, 7))
dend = shc.dendrogram(shc.linkage(data, method=linkage, metric = distance))


# %%
cluster = AgglomerativeClustering(n_clusters=k, affinity=distance, linkage=linkage)
cluster.fit(data)
labels = cluster.labels_

raw["labels"] = labels

data.plot.scatter(x = "concepts", y = "objects", c = list(labels), cmap="Set1", figsize=(10,7), sharex = False)


#%%

#fig = plt.figure()

#ax = plt.axes(projection='3d')
#ax.scatter(data["concepts"],data["objects"],data["units"], c=list(labels), cmap='Set1')
#ax.set_xlabel('Concepts')
#ax.set_ylabel('Objects')
#ax.set_zlabel('Units')
# %%

#%%
metr = {"sil": [], "ch": [], "k": []}
for k in range(2,15):
    cluster = AgglomerativeClustering(n_clusters=k, affinity=distance, linkage=linkage)
    cluster.fit(data)
    labels = cluster.labels_


    metr["sil"].append(metrics.silhouette_score(data, labels, metric=distance))
    metr["ch"].append(metrics.calinski_harabasz_score(data, labels))
    metr["k"].append(k)

# %%
metr = pd.DataFrame(metr)
#%%
metr[["sil", "k"]].plot(x = "k")
# %%
metr[["ch", "k"]].plot(x = "k")


# %%
test_results = pd.read_csv("/workspaces/speechrecognition/data/interim/test_scores.csv", index_col="teacher")


test_results["pre"] = test_results["pre"] / 18
test_results["post"] = test_results["post"] / 36
test_results["change"] = test_results["post"] - test_results["pre"]
# %%
df = pd.merge(raw, test_results, left_index = True, right_index = True)

#%%
# %%
df.groupby(by="labels").count()
# %%
df.groupby(by="labels").mean()
#%%
df.groupby(by="labels").median()
#%%
df.groupby(by="labels").std()


# %%
df["objects"].hist(bins = 20, cumulative = True)

# %%
clf = tree.DecisionTreeClassifier()
dummy_clf = DummyClassifier(strategy="uniform")
print("Decision tree: " + str(cross_val_score(clf, X, y, cv=5, n_jobs=4).mean()))

print("Random: " + str(cross_val_score(dummy_clf, X, y, cv=5, n_jobs=4).mean()))
# %%
plt.figure()
clf.fit(X,y)
tree.plot_tree(clf, filled = True)
plt.show()
# %%
