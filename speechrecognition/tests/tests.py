#%%
from load_asr_data import load_keywords, load_transcripts, load_test_transcript
import itertools as it
from network_analysis_weighted import get_edges, rolling_window, stem_transcript, edge_to_tnet
#%%
#TODO TEST stem_transcript function
#TODO TEST window function
def test_get_edges():
    test_transcript = [["a","b","c","d","e","f"],
            ["j","b","c","e","e","h"],
            ["j","t","t","e","e","h"],
            ["a","b","c","d","e","f"]]
    test_keywords = ["a","b","f"]
    assert get_edges(test_transcript, test_keywords) == [('a', 'b'), ('a', 'f'), ('b', 'f'), ('a', 'b'), ('a', 'f'), ('b', 'f'), ('a', 'b'), ('b', 'b'), ('b', 'f')]

def test_edge_to_tnet():
    test_edge1 = ('teho', 'työ', 3)
    test_edge2 = ('teho', 'verkkojännit', 1)
    assert edge_to_tnet(test_edge1) == [{"from":"teho", "to":"työ", "weight":3},{"from":"työ", "to":"teho", "weight":3}]
    assert edge_to_tnet(test_edge2) == [{"from":"teho", "to":"verkkojännit", "weight":1},{"from":"verkkojännit", "to":"teho", "weight":1}]
#%%
test_get_edges()
test_edge_to_tnet()