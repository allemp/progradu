#%%
import pytest
import physicsasr.features.create_features as cf

def test_keyword_counter():
    keywords1 = ["cat","bat", "dog"]
    words1 = ["the", "cat", "is", "not", "a", "dog"]

    keywords2 = ["cat","bat", "dog"]
    words2 = ["the", "rooster", "is", "not", "a", "hen"]

    keywords3 = ["cat", "dog"]
    words3 = ['cat', 'bat', 'dog', 'the', 'cat', 'is', 'not', 'a', 'dog', 'dog']

    assert cf.keyword_counter(words1, keywords1) == 2
    assert cf.keyword_counter(words2, keywords2) == 0
    assert cf.keyword_counter(words3, keywords3) == 5


def test_fixed_window():
    case1 = [[1],[2],[3],[4],[5],[6],[7],[8],[9]]
    case2 = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]

    assert list(cf.fixed_window(case1, 2)) == [[1,2],[3,4],[5,6],[7,8],[9]]
    assert list(cf.fixed_window(case2, 2)) == [[1,2],[3,4],[5,6],[7,8],[9,10]]
    assert list(cf.fixed_window(case1, 3)) == [[1,2,3],[4,5,6],[7,8,9]]

def test_transcript_keyword_freq():
    case1 = [["cat","bat", "dog"], ["the", "cat", "is", "not", "a", "dog"]]
    keywords1 = ["cat", "dog"]
    assert cf.transcript_keyword_freq(case1,keywords1,1)["freq_keywords"] == [2,2]
    assert cf.transcript_keyword_freq(case1,keywords1,2)["freq_keywords"] == [4]
#%%
def test_matrix_profile():
    assert abs(cf.matrix_profile([1,2,3,4,5,6,7,8,9,1,2,3,11,12,13,14],
            4)[0][0] - 4.21468485e-08) < 0.1