# -*- coding: utf-8 -*-
"""
Various utility functions

Note: This was more or less copied over from Spotlight
"""
import logging

import numpy as np
from scipy.sparse import coo_matrix
from scipy.stats import kendalltau, spearmanr, normaltest
import torch


_logger = logging.getLogger(__name__)


def is_cuda_available():
    return torch.cuda.is_available()


def gpu(tensor, gpu=False):

    if gpu:
        return tensor.cuda()
    else:
        return tensor


def cpu(tensor):

    if tensor.is_cuda:
        return tensor.cpu()
    else:
        return tensor


def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', 128)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    random_state = kwargs.get('random_state')

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    if random_state is None:
        random_state = np.random.RandomState()

    shuffle_indices = np.arange(len(arrays[0]))
    random_state.shuffle(shuffle_indices)

    if len(arrays) == 1:
        return arrays[0][shuffle_indices]
    else:
        return tuple(x[shuffle_indices] for x in arrays)


def assert_no_grad(variable):

    if variable.requires_grad:
        raise ValueError(
            "nn criterions don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )


def set_seed(seed, cuda=False):

    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)


def sample_items(num_items, shape, random_state=None):
    """
    Randomly sample a number of items.

    Parameters
    ----------

    num_items: int
        Total number of items from which we should sample:
        the maximum value of a sampled item id will be smaller
        than this.
    shape: int or tuple of ints
        Shape of the sampled array.
    random_state: np.random.RandomState instance, optional
        Random state to use for sampling.

    Returns
    -------

    items: np.array of shape [shape]
        Sampled item ids.
    """

    if random_state is None:
        random_state = np.random.RandomState()

    items = random_state.randint(0, num_items, shape, dtype=np.int64)

    return items


def process_ids(user_ids, item_ids, n_items, use_cuda, cartesian):
    if item_ids is None:
        item_ids = np.arange(n_items, dtype=np.int64)

    if np.isscalar(user_ids):
        user_ids = np.array(user_ids, dtype=np.int64)

    user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
    item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

    if cartesian:
        item_ids, user_ids = (item_ids.repeat(user_ids.size(0), 1),
                              user_ids.repeat(1, item_ids.size(0)).view(-1, 1))
    else:
        user_ids = user_ids.expand(item_ids.size(0), 1)

    user_var = gpu(user_ids, use_cuda)
    item_var = gpu(item_ids, use_cuda)

    return user_var.squeeze(), item_var.squeeze()


def get_cov(X, Y, use_zero_mean=False):
    """
    Returns covariance between data points of two arrays

    Args:
        X:
        Y:
        use_zero_mean:

    Returns:

    """
    result = np.zeros((2, 2))

    if use_zero_mean:
        result[0, 0] = np.mean(X * X)
        result[1, 1] = np.mean(Y * Y)
        result[1, 0] = result[0, 1] = np.mean(X * Y)
    else:
        result = np.cov(np.vstack([X, Y]))

    return result


def get_corr_coef(X, Y, corr_type='pearson') -> float:
    """
    Returns the correlation coefficient for data points of two array
    Defaults to Pearson Correlation

    Args:
        X:
        Y:
        corr_type:

    Returns:

    """
    if corr_type == 'pearson':
        return np.corrcoef(X, Y)[0, 1]
    elif corr_type == 'kendalltau':
        return kendalltau(np.argsort(X), np.argsort(Y))[0]
    elif corr_type == 'spearman':
        return spearmanr(X, Y)[0]


def get_entity_corr_coef(interactions: coo_matrix, entity_id: int, entity_type: str,
                         embeddings: dict,
                         ignore_sparse_zeros=True, use_zero_mean=False,
                         corr_type='pearson', neg_sampling=False, check_normal_dist=True):
    """
    Assumes a rating matrix with rows for users and columns for items
    """
    p = embeddings['user'].shape[1]
    cov_for_p_variables = []

    if entity_type == 'user':
        embed = embeddings['user'][entity_id]
        # embedding used for covariance computation
        cov_embed = embeddings['item']
        # ratings used for covariance computation
        ratings = np.squeeze(np.asarray(interactions.tocsr()[entity_id, :].todense()))
    elif entity_type == 'item':
        embed = embeddings['item'][entity_id]
        # embedding used for covariance computation
        cov_embed = embeddings['user']
        # ratings used for covariance computation
        ratings = np.squeeze(np.asarray(interactions.tocsr()[:, entity_id].todense()))

    if ignore_sparse_zeros:
        pos_idx = np.where(ratings != 0)[0]
        pos_ratings = ratings[pos_idx]

    # TODO: Use `sample_items` method
    # Use this for BPR
    if neg_sampling:
        if entity_type == 'user':
            n_sample = interactions.shape[1]
        else:
            n_sample = interactions.shape[0]
        neg_idx = np.random.randint(n_sample, size=len(pos_idx))
        # neg_idx = np.random.choice(np.setdiff1d(np.arange(interactions.n_items),
        #                                         pos_idx), size=len(pos_idx),
        #                            replace=False)
        neg_ratings = [0] * len(pos_ratings)
        idx = np.concatenate([pos_idx, neg_idx])
        ratings = np.concatenate([pos_ratings, neg_ratings])

    cov_embed = cov_embed[idx]

    for k in range(p):
        cov_embed_latent_variables_at_k = cov_embed[:, k]
        cov_mat_for_k = get_cov(ratings, cov_embed_latent_variables_at_k,
                                use_zero_mean=use_zero_mean)
        cov_for_k = cov_mat_for_k[0, 1]
        cov_for_p_variables.append(cov_for_k)

    # TODO: Change from printing back to logging
    if check_normal_dist:
        alpha = 1e-3
        p_embed = normaltest(embed)[1]
        p_cov_for_p_variables = normaltest(cov_for_p_variables)[1]
        if p_embed < alpha:
            print(
                f"{entity_type}-{entity_id}: Entity Embeddings are unlikely normally distributed.")
        if p_cov_for_p_variables < alpha:
            print(
                f"{entity_type}-{entity_id}: Covariances are unlikely normally distributed.")

    cov_for_p_variables = np.array(cov_for_p_variables)
    corr_coef = get_corr_coef(embed, cov_for_p_variables, corr_type=corr_type)

    return corr_coef
