from lrann import datasets

import pytest


def test_get_movielens_100k(tmp_path):
    loader = datasets.Loader(data_dir=tmp_path)
    interactions = loader.load_movielens('100k')
    df = interactions.topandas()
    assert df.shape == (100836, 3)


def test_get_movielens_100k_old(tmp_path):
    loader = datasets.Loader(data_dir=tmp_path)
    interactions = loader.load_movielens('100k-old')
    df = interactions.topandas()
    assert df.shape == (100000, 3)


@pytest.mark.skip
def test_get_movielens_20m(tmp_path):
    loader = datasets.Loader(data_dir=tmp_path)
    interactions = loader.load_movielens('20m')
    df = interactions.topandas()
    assert df.shape == (20000263, 3)
