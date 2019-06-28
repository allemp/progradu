require(tnet)
raw <- read.csv("data/tnet_data.csv", encoding="UTF-8")
df <- raw
df$from <- as.numeric(df$from)
df$to <- as.numeric(df$to)
undirected.net <- as.tnet(df)


degree_w(undirected.net)
betweenness_w(undirected.net)
closeness_w(undirected.net)
clustering_w(undirected.net)
