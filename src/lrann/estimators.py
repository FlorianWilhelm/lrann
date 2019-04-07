# -*- coding: utf-8 -*-
"""
Functionality related to estimators which train a model

Note: The Implicit Est class is more or less FMModel from Spotlight
"""
from abc import ABCMeta, abstractmethod
import numpy as np

import torch
import torch.optim as optim

from .losses import bpr_loss, logistic_loss
from .utils import (cpu, gpu, minibatch, set_seed, shuffle, sample_items,
                    process_ids)


class BaseEstimator(metaclass=ABCMeta):
    def __init__(self, *, model, use_cuda=False, random_state=None):
        self._model = model
        self._model = gpu(model, use_cuda)
        self._use_cuda = use_cuda
        self._random_state = random_state or np.random.RandomState()

        set_seed(self._random_state.randint(-10 ** 8, 10 ** 8),
                 cuda=self._use_cuda)

    def __repr__(self):
        return '<{}: {}>'.format(
            self.__class__.__name__,
            repr(self._model),
        )

    @abstractmethod
    def fit(self, interactions, **kwargs):
        pass

    def predict(self, user_ids, item_ids=None, cartesian=False):
        """
        Make predictions: given a user id, compute the recommendation
        scores for items.

        Use this as Mixin to avoid double implementation

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
        cartesian: bool, optional
            Calculate the prediction for each item times each user

        Returns
        -------

        predictions: np.array
            Predicted scores for all items in item_ids.
        """
        self._model.train(False)

        if np.isscalar(user_ids):
            n_users = 1
        else:
            n_users = len(user_ids)
        user_ids, item_ids = process_ids(user_ids, item_ids,
                                         self._model.n_items,
                                         self._use_cuda,
                                         cartesian)

        out = self._model(user_ids, item_ids)
        out = cpu(out).detach().numpy()
        if cartesian:
            return out.reshape(n_users, -1)
        else:
            return out.flatten()


class ImplicitEst(BaseEstimator):
    """Estimator for implicit feedback using BPR
    """
    def __init__(self,
                 *,
                 model,
                 embedding_dim=32,
                 n_iter=10,
                 batch_size=128,
                 l2=0.0,
                 learning_rate=1e-2,
                 optimizer=None,
                 n_negative_samples=5,
                 **kwargs):
        super().__init__(model=model, **kwargs)
        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2 = l2
        self._n_negative_samples = n_negative_samples

        if optimizer is None:
            self._optimizer = optim.Adam(
                self._model.parameters(),
                weight_decay=self._l2,
                lr=self._learning_rate
            )
        else:
            self._optimizer = self._optimizer(self._model.parameters())

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
        self._model.train(True)

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
            minibatch_num = -1
            batches = minibatch(user_ids_tensor,
                                item_ids_tensor,
                                batch_size=self._batch_size)
            for minibatch_num, (batch_user, batch_item) in enumerate(batches):
                positive_prediction = self._model(batch_user, batch_item)
                negative_prediction = self._get_negative_prediction(batch_user)

                self._optimizer.zero_grad()

                loss = bpr_loss(positive_prediction, negative_prediction)
                epoch_loss += loss.item()

                loss.backward()
                self._optimizer.step()

            if minibatch_num == -1:
                raise RuntimeError("There is not even a single mini-batch to train on!")

            epoch_loss /= minibatch_num + 1

            if verbose:
                print('Epoch {}: loss {}'.format(epoch_num, epoch_loss))

            if np.isnan(epoch_loss) or epoch_loss == 0.0:
                raise ValueError('Degenerate epoch loss: {}'
                                 .format(epoch_loss))

    def _get_negative_prediction(self, user_ids):

        negative_items = sample_items(
            self._model.n_items,
            len(user_ids),
            random_state=self._random_state)
        negative_var = gpu(torch.from_numpy(negative_items), self._use_cuda)
        negative_prediction = self._model(user_ids, negative_var)

        return negative_prediction


class ExplicitEst(BaseEstimator):
    """
    An explicit feedback matrix factorization model. Uses a classic
    matrix factorization [1]_ approach, with latent vectors used
    to represent both users and items. Their dot product gives the
    predicted score for a user-item pair.
    The latent representation is given by
    :class:`spotlight.factorization.representations.BilinearNet`.
    .. [1] Koren, Yehuda, Robert Bell, and Chris Volinsky.
       "Matrix factorization techniques for recommender systems."
       Computer 42.8 (2009).
    Parameters
    ----------
    loss: string, optional
        One of 'regression', 'poisson', 'logistic'
        corresponding to losses from :class:`spotlight.losses`.
    embedding_dim: int, optional
        Number of embedding dimensions to use for users and items.
    n_iter: int, optional
        Number of iterations to run.
    batch_size: int, optional
        Minibatch size.
    l2: float, optional
        L2 loss penalty.
    learning_rate: float, optional
        Initial learning rate.
    optimizer_func: function, optional
        Function that takes in module parameters as the first argument and
        returns an instance of a PyTorch optimizer. Overrides l2 and learning
        rate if supplied. If no optimizer supplied, then use ADAM by default.
    use_cuda: boolean, optional
        Run the model on a GPU.
    representation: a representation module, optional
        If supplied, will override default settings and be used as the
        main network module in the model. Intended to be used as an escape
        hatch when you want to reuse the model's training functions but
        want full freedom to specify your network topology.
    sparse: boolean, optional
        Use sparse gradients for embedding layers.
    random_state: instance of numpy.random.RandomState, optional
        Random state to use when fitting.
    """

    def __init__(self,
                 *,
                 model,
                 embedding_dim=32,
                 n_iter=10,
                 batch_size=256,
                 l2=0.0,
                 learning_rate=1e-2,
                 optimizer=None,
                 **kwargs):

        super().__init__(model=model, **kwargs)

        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2 = l2
        if optimizer is None:
            self._optimizer = optim.Adam(
                self._model.parameters(),
                weight_decay=self._l2,
                lr=self._learning_rate
            )
        else:
            self._optimizer = self._optimizer(self._model.parameters())

    def fit(self, interactions, verbose=False):
        """
        Fit the model.
        When called repeatedly, model fitting will resume from
        the point at which training stopped in the previous fit
        call.
        Parameters
        ----------
        interactions: :class:`spotlight.interactions.Interactions`
            The input dataset. Must have ratings.
        verbose: bool
            Output additional information about current epoch and loss.
        """
        user_ids = interactions.user_ids.astype(np.int64)
        item_ids = interactions.item_ids.astype(np.int64)

        for epoch_num in range(self._n_iter):

            users, items, ratings = shuffle(user_ids,
                                            item_ids,
                                            interactions.ratings,
                                            random_state=self._random_state)

            user_ids_tensor = gpu(torch.from_numpy(users),
                                  self._use_cuda)
            item_ids_tensor = gpu(torch.from_numpy(items),
                                  self._use_cuda)
            ratings_tensor = gpu(torch.from_numpy(ratings),
                                 self._use_cuda)

            epoch_loss = 0.0
            minibatch_num = -1
            batches = minibatch(user_ids_tensor,
                                item_ids_tensor,
                                ratings_tensor,
                                batch_size=self._batch_size)
            for minibatch_num, (batch_user, batch_item, batch_ratings) in enumerate(batches):

                predictions = self._model(batch_user, batch_item)

                self._optimizer.zero_grad()

                loss = logistic_loss(batch_ratings, predictions.double())
                epoch_loss += loss.item()

                loss.backward()
                self._optimizer.step()

            epoch_loss /= minibatch_num + 1

            if verbose:
                print('Epoch {}: loss {}'.format(epoch_num, epoch_loss))

            if np.isnan(epoch_loss) or epoch_loss == 0.0:
                raise ValueError('Degenerate epoch loss: {}'
                                 .format(epoch_loss))
