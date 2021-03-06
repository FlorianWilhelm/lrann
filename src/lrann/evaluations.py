# -*- coding: utf-8 -*-
"""
Functions to evaluate a trained model

Note: The file was more or less taken from Spotlight
"""

import numpy as np

import scipy.stats as st


FLOAT_MAX = np.finfo(np.float32).max


def mrr_score(model, test, train=None):
    """
    Compute mean reciprocal rank (MRR) scores. One score
    is given for every user with interactions in the test
    set, representing the mean reciprocal rank of all their
    test items.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, scores of known
        interactions will be set to very low values and so not
        affect the MRR.

    Returns
    -------

    mrr scores: numpy array of shape (num_users,)
        Array of MRR scores for each user in test.
    """

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    mrrs = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)

        if train is not None:
            predictions[train[user_id].indices] = FLOAT_MAX

        mrr = (1.0 / st.rankdata(predictions)[row.indices]).mean()

        mrrs.append(mrr)

    return np.array(mrrs)


def _get_precision_recall(predictions, targets, k):

    predictions = predictions[:k]
    n_hit = len(set(predictions).intersection(set(targets)))

    return float(n_hit) / len(predictions), float(n_hit) / len(targets)


def precision_recall_score(model, test, train=None, k=10):
    """
    Compute Precision@k and Recall@k scores. One score
    is given for every user with interactions in the test
    set, representing the Precision@k and Recall@k of all their
    test items.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, scores of known
        interactions will not affect the computed metrics.
    k: int or array of int,
        The maximum number of predicted items
    Returns
    -------

    (Precision@k, Recall@k): numpy array of shape (num_users, len(k))
        A tuple of Precisions@k and Recalls@k for each user in test.
        If k is a scalar, will return a tuple of vectors. If k is an
        array, will return a tuple of arrays, where each row corresponds
        to a user and each column corresponds to a value of k.
    """

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    if np.isscalar(k):
        k = np.array([k])

    precision = []
    recall = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)

        if train is not None:
            rated = train[user_id].indices
            predictions[rated] = FLOAT_MAX

        predictions = predictions.argsort()

        targets = row.indices

        user_precision, user_recall = zip(*[
            _get_precision_recall(predictions, targets, x)
            for x in k
        ])

        precision.append(user_precision)
        recall.append(user_recall)

    precision = np.array(precision).squeeze()
    recall = np.array(recall).squeeze()

    return precision, recall


def auc_score(model, test, train=None, auc_selection_seed=42):
    """
    See https://arxiv.org/pdf/1508.06091.pdf

    Args:
        model:
        test:
        train:
        auc_selection_seed:

    Returns:

    """
    # TODO: Implement known positive removal (not urgent as not applicable for Movielens)
    test = test.tocsr()
    np.random.seed(auc_selection_seed)

    auc_score = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        # Make predictions for all items
        predictions = model.predict(user_id)

        pos_targets = row.indices
        n_preds = len(pos_targets)
        neg_targets = np.setdiff1d(np.arange(len(predictions)), pos_targets)
        neg_targets = np.random.choice(neg_targets, size=n_preds, replace=False)

        # Obtain predictions for all positives
        pos_predictions = predictions[pos_targets]

        # Obtain predictions for random set of unobserved that has the same length
        # as the positives
        neg_predictions = predictions[neg_targets]

        # Compare both ratings for ranking distortions, i.e. positive < negative
        user_auc_score = (pos_predictions > neg_predictions).sum()/n_preds

        auc_score.append(user_auc_score)

    return np.array(auc_score)


def rmse_score(model, test):
    """
    Compute RMSE score for test interactions.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.

    Returns
    -------

    rmse_score: float
        The RMSE score.
    """

    predictions = model.predict(test.user_ids, test.item_ids)
    ratings = np.clip(test.ratings, 0, 1)  # bring -1 to 0

    return np.sqrt(((ratings - predictions) ** 2).mean())
