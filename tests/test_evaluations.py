# -*- coding: utf-8 -*-
from lrann.estimators import ImplicitEst, ExplicitEst
from lrann.models import BilinearNet
from lrann.datasets import DataLoader, random_train_test_split
from lrann.evaluations import precision_recall_score, mrr_score


def test_metrics():
    data = DataLoader().load_movielens('100k')
    train, test = random_train_test_split(data)
    lra_model = BilinearNet(data.n_users, data.n_items, embedding_dim=32, sparse=False)
    lra_est = ImplicitEst(model=lra_model,
                          n_iter=1)
    lra_est.fit(train, verbose=True)
    precision_recall_score(lra_est, test)
    mrr_score(lra_est, test)
