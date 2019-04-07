# -*- coding: utf-8 -*-
import numpy as np
from lrann.estimators import ImplicitEst, ExplicitEst
from lrann.models import BilinearNet
from lrann.datasets import DataLoader, random_train_test_split
import pytest


def test_implicit_est_lra():
    data = DataLoader().load_movielens('100k')
    train, test = random_train_test_split(data)
    lra_model = BilinearNet(data.n_users, data.n_items, embedding_dim=32, sparse=False)
    lra_est = ImplicitEst(model=lra_model,
                          n_iter=1)
    lra_est.fit(train, verbose=True)
    user_ids = np.arange(5)
    item_ids = np.arange(5)
    assert lra_est.predict(1, item_ids, cartesian=False).shape == (5,)
    assert lra_est.predict(1, item_ids, cartesian=True).shape == (1, 5)
    assert lra_est.predict(user_ids, item_ids, cartesian=False).shape == (5,)
    assert lra_est.predict(user_ids, item_ids, cartesian=True).shape == (5, 5)
    with pytest.raises(RuntimeError):
        lra_est.predict(user_ids, np.arange(6), cartesian=False)


def test_explicit_est_lra():
    data = DataLoader().load_movielens('100k')
    data.binarize_(4)
    train, test = random_train_test_split(data)
    lra_model = BilinearNet(data.n_users, data.n_items, embedding_dim=32, sparse=False)
    lra_est = ExplicitEst(model=lra_model,
                          n_iter=1)
    lra_est.fit(train, verbose=True)
    item_ids = np.arange(5)
    assert lra_est.predict(1, item_ids, cartesian=False).shape == (5,)
