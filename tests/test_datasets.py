# -*- coding: utf-8 -*-
from lrann import datasets

import pytest
import numpy as np
from numpy.testing import assert_array_equal


def test_get_movielens_100k(tmp_path):
    loader = datasets.DataLoader(data_dir=tmp_path)
    interactions = loader.load_movielens('100k')
    df = interactions.topandas()
    assert df.shape == (100836, 3)


def test_get_movielens_100k_old(tmp_path):
    loader = datasets.DataLoader(data_dir=tmp_path)
    interactions = loader.load_movielens('100k-old')
    df = interactions.topandas()
    assert df.shape == (100000, 3)


@pytest.mark.skip
def test_get_movielens_20m(tmp_path):
    loader = datasets.DataLoader(data_dir=tmp_path)
    interactions = loader.load_movielens('20m')
    df = interactions.topandas()
    assert df.shape == (20000263, 3)


def test_implicit_interactions(interactions1):
    interactions1.implicit_(4)
    assert len(interactions1) == 3
    assert_array_equal(interactions1.user_ids, np.array([1, 0, 2]))


def test_binarize_interactions(interactions1):
    interactions1.binarize_(4)
    assert len(interactions1) == 5
    assert_array_equal(interactions1.ratings, np.array([-1, -1, 1, 1, 1]))
