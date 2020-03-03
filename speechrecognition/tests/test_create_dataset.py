import pytest

import physicsasr.dataset.create_dataset as cd

def test_nltk_stemmer():
    case1 = ["apina", "banaani", "kivi"]
    result1 = ["ap", "bana", "kivi"]

    case2 = ["apinaa", "banaania", "kiveä"]
    result2 = ["apin", "banaan", "kive"]

    assert cd.nltk_stemmer(case1) == result1
    assert cd.nltk_stemmer(case2) == result2

def test_voikko_lemmatizer():
    case1 = ["apina", "banaani", "kivi"]
    result1 = ["apina", "banaani", "kivi"]

    case2 = ["apinaa", "banaania", "kiveä"]
    result2 = ["apina", "banaani", "kivetä"]
    assert cd.voikko_lemmatizer(case1) == result1
    assert cd.voikko_lemmatizer(case2) == result2

def test_create_keywordlist():
    case1 = ["apina\n", "banaani\n", "kivi\n","\n"]
    result1 = ["ap", "bana", "kivi", ""]
    
    case2 = ["apinaa\n", "banaania\n", "kiveä\n","\n"]
    result2 = ["apina", "banaani", "kivetä"]

    assert cd.create_keywordlist(case1, cd.nltk_stemmer) == result1
    assert cd.create_keywordlist(case2, cd.voikko_lemmatizer) == result2

def test_create_transcript():
    case1 = ["0.0 5.0 \n","0.5 10.0 apina banaani kivi\n", "10.0 15.0 apinaa banaania kiveä\n"]
    result1 = [[""], ["ap", "bana", "kivi"], ["apin", "banaan", "kive"]]

    case2 = ["0.0 5.0 \n","0.5 10.0 apina banaani kivi\n", "10.0 15.0 apinaa banaania kiveä\n"]
    result2 = [[], ["apina", "banaani", "kivi"], ["apina", "banaani", "kivetä"]]

    assert cd.create_transcript(case1, cd.nltk_stemmer) == result1
    assert cd.create_transcript(case2, cd.voikko_lemmatizer) == result2