P=python3
KWPATH=data/raw/keywords/master_keywords_unstemmed.txt
TRPATH=data/raw/transcripts
TSPATH=data/raw/test_results/CKPT_1dim_mit_Anker.sav
graphs := $(patsubst reports/figures/%.dot,reports/figures/%.png,$(wildcard reports/figures/*.dot))

all: data/interim/keywords_stemmed.json data/interim/keywords_voikko.json data/interim/transcripts_voikko.json data/interim/transcripts_stemmed.json data/interim/test_scores.csv $(graphs)

data/interim/keywords_stemmed.json: physicsasr/dataset/create_dataset.py
	$(P) $^ --path=$(KWPATH) --type=keywords --baseformer=nltk > $@

data/interim/keywords_voikko.json: physicsasr/dataset/create_dataset.py
	$(P) $^ --path=$(KWPATH) --type=keywords --baseformer=voikko > $@

data/interim/transcripts_voikko.json: physicsasr/dataset/create_dataset.py
	$(P) $^ --path=$(TRPATH) --type=transcripts --baseformer=voikko > $@

data/interim/transcripts_stemmed.json: physicsasr/dataset/create_dataset.py
	$(P) $^ --path=$(TRPATH) --type=transcripts --baseformer=nltk > $@

data/interim/test_scores.csv: physicsasr/dataset/create_test_scores.R
	Rscript --vanilla $^ $(TSPATH) $@

reports/figures/%.png: reports/figures/%.dot
	dot $< -Tpng -o $@