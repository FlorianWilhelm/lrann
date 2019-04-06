# -*- coding: utf-8 -*-
"""
Functionality related to training a given model architecture

Note: The Trainer class is more or less FMModel from Spotlight
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import (cpu, gpu, minibatch, set_seed, shuffle, sample_items,
                    process_ids)


class Trainer(object):
    def __init__(self,
                 *,
                 num_users,
                 num_items,
                 net,
                 embedding_dim=1,
                 n_iter=10,
                 batch_size=128,
                 l2=0.0,
                 learning_rate=1e-2,
                 optimizer_func=None,
                 use_cuda=False,
                 sparse=False,
                 random_state=None,
                 num_negative_samples=5):

        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2 = l2
        self._use_cuda = use_cuda
        self._sparse = sparse
        self._random_state = random_state or np.random.RandomState()
        self._num_negative_samples = num_negative_samples
        self._net = net
        self._num_items = num_items
        self._num_users = num_users
        self._optimizer = optim.Adam(
            self._net.parameters(),
            weight_decay=self._l2,
            lr=self._learning_rate
        )

        set_seed(self._random_state.randint(-10 ** 8, 10 ** 8),
                 cuda=self._use_cuda)

    def _loss(self, positive_predictions, negative_predictions, mask=None):
        loss = (1.0 - torch.sigmoid(positive_predictions -
                                    negative_predictions))

        if mask is not None:
            mask = mask.float()
            loss = loss * mask
            return loss.sum() / mask.sum()

        return loss.mean()

    def fit(self, interactions, verbose=False):
        """
        Fit the model.

        When called repeatedly, model fitting will resume from
        the point at which training stopped in the previous fit
        call.

        Parameters
        ----------

        interactions: :class:`spotlight.interactions.Interactions`
            The input dataset.

        verbose: bool
            Output additional information about current epoch and loss.
        """
        self._net.train(True)

        user_ids = interactions.user_ids.astype(np.int64)
        item_ids = interactions.item_ids.astype(np.int64)

        for epoch_num in range(self._n_iter):

            users, items = shuffle(user_ids,
                                   item_ids,
                                   random_state=self._random_state)

            user_ids_tensor = gpu(torch.from_numpy(users),
                                  self._use_cuda)
            item_ids_tensor = gpu(torch.from_numpy(items),
                                  self._use_cuda)

            epoch_loss = 0.0
            a = gpu(torch.from_numpy(np.array([user_ids[10]])), self._use_cuda)
            # print(self._net.user_embeddings(a))

            for (minibatch_num,
                 (batch_user,
                  batch_item)) in enumerate(minibatch(user_ids_tensor,
                                                      item_ids_tensor,
                                                      batch_size=self._batch_size)):
                positive_prediction = self._net(batch_user, batch_item)
                negative_prediction = self._get_negative_prediction(batch_user)

                self._optimizer.zero_grad()

                loss = self._loss(positive_prediction, negative_prediction)
                # for scaled
                # loss += 0.01 * torch.norm(self._net.scale - 1., 2)
                epoch_loss += loss.item()

                loss.backward()
                self._optimizer.step()

            epoch_loss /= minibatch_num + 1

            if verbose:
                print('Epoch {}: loss {}'.format(epoch_num, epoch_loss))

            if np.isnan(epoch_loss) or epoch_loss == 0.0:
                raise ValueError('Degenerate epoch loss: {}'
                                 .format(epoch_loss))

    def _get_negative_prediction(self, user_ids):

        negative_items = sample_items(
            self._num_items,
            len(user_ids),
            random_state=self._random_state)  # np.random.RandomState(42))#
        negative_var = gpu(torch.from_numpy(negative_items), self._use_cuda)
        negative_prediction = self._net(user_ids, negative_var)

        return negative_prediction

    def predict(self, user_ids, item_ids=None):
        """
        Make predictions: given a user id, compute the recommendation
        scores for items.

        Parameters
        ----------

        user_ids: int or array
           If int, will predict the recommendation scores for this
           user for all items in item_ids. If an array, will predict
           scores for all (user, item) pairs defined by user_ids and
           item_ids.
        item_ids: array, optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.

        Returns
        -------

        predictions: np.array
            Predicted scores for all items in item_ids.
        """
        self._net.train(False)

        user_ids, item_ids = process_ids(user_ids, item_ids,
                                         self._num_items,
                                         self._use_cuda)

        out = self._net(user_ids, item_ids)

        return cpu(out).detach().numpy().flatten()

