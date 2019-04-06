# -*- coding: utf-8 -*-
"""
Different data sets

Note: Some parts copied over from Spotlight
"""

import os
import requests

import h5py
import numpy as np
import scipy.sparse as sp


DATA_DIR = os.path.join(os.path.expanduser('~'), 'recsys_data')


URL_PREFIX = ('https://github.com/maciejkula/recommender_datasets/'
              'releases/download')
VERSION = 'v0.2.0'


def create_data_dir(path):

    if not os.path.isdir(path):
        os.makedirs(path)


def download(url, dest_path, data_dir=DATA_DIR):
    req = requests.get(url, stream=True)
    req.raise_for_status()

    with open(dest_path, 'wb') as fd:
        for chunk in req.iter_content(chunk_size=2**20):
            fd.write(chunk)


def get_data(url, dest_subdir, dest_filename, download_if_missing=True):
    data_dir = os.path.join(os.path.abspath(DATA_DIR), dest_subdir)
    create_data_dir(data_dir)
    dest_path = os.path.join(data_dir, dest_filename)

    if not os.path.isfile(dest_path):
        if download_if_missing:
            download(url, dest_path)
        else:
            raise IOError('Dataset missing.')

    return dest_path


VARIANTS = ('100K',
            '1M',
            '10M',
            '20M')


def _get_movielens(dataset):

    extension = '.hdf5'

    path = get_data('/'.join((URL_PREFIX,
                              VERSION,
                              dataset + extension)),
                    os.path.join('movielens', VERSION),
                    'movielens_{}{}'.format(dataset,
                                            extension))

    with h5py.File(path, 'r') as data:
        return (data['/user_id'][:],
                data['/item_id'][:],
                data['/rating'][:],
                data['/timestamp'][:])


def get_movielens_dataset(variant='100K'):
    """
    Download and return one of the Movielens datasets.

    Parameters
    ----------

    variant: string, optional
         String specifying which of the Movielens datasets
         to download. One of ('100K', '1M', '10M', '20M').

    Returns
    -------

    Interactions: :class:`spotlight.interactions.Interactions`
        instance of the interactions class
    """

    if variant not in VARIANTS:
        raise ValueError('Variant must be one of {}, '
                         'got {}.'.format(VARIANTS, variant))

    url = 'movielens_{}'.format(variant)

    return Interactions(*_get_movielens(url))


def _sliding_window(tensor, window_size, step_size=1):

    for i in range(len(tensor), 0, -step_size):
        yield tensor[max(i - window_size, 0):i]


def _generate_sequences(user_ids, item_ids,
                        indices,
                        max_sequence_length,
                        step_size):

    for i in range(len(indices)):

        start_idx = indices[i]

        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]

        for seq in _sliding_window(item_ids[start_idx:stop_idx],
                                   max_sequence_length,
                                   step_size):

            yield (user_ids[i], seq)


class Interactions(object):
    """
    Interactions object. Contains (at a minimum) pair of user-item
    interactions, but can also be enriched with ratings, timestamps,
    and interaction weights.

    For *implicit feedback* scenarios, user ids and item ids should
    only be provided for user-item pairs where an interaction was
    observed. All pairs that are not provided are treated as missing
    observations, and often interpreted as (implicit) negative
    signals.

    For *explicit feedback* scenarios, user ids, item ids, and
    ratings should be provided for all user-item-rating triplets
    that were observed in the dataset.

    Parameters
    ----------

    user_ids: array of np.int32
        array of user ids of the user-item pairs
    item_ids: array of np.int32
        array of item ids of the user-item pairs
    ratings: array of np.float32, optional
        array of ratings
    timestamps: array of np.int32, optional
        array of timestamps
    weights: array of np.float32, optional
        array of weights
    num_users: int, optional
        Number of distinct users in the dataset.
        Must be larger than the maximum user id
        in user_ids.
    num_items: int, optional
        Number of distinct items in the dataset.
        Must be larger than the maximum item id
        in item_ids.

    Attributes
    ----------

    user_ids: array of np.int32
        array of user ids of the user-item pairs
    item_ids: array of np.int32
        array of item ids of the user-item pairs
    ratings: array of np.float32, optional
        array of ratings
    timestamps: array of np.int32, optional
        array of timestamps
    weights: array of np.float32, optional
        array of weights
    num_users: int, optional
        Number of distinct users in the dataset.
    num_items: int, optional
        Number of distinct items in the dataset.
    """

    def __init__(self, user_ids, item_ids,
                 ratings=None,
                 timestamps=None,
                 weights=None,
                 num_users=None,
                 num_items=None):

        self.num_users = num_users or int(user_ids.max() + 1)
        self.num_items = num_items or int(item_ids.max() + 1)

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.timestamps = timestamps
        self.weights = weights

        self._check()

    def __repr__(self):

        return ('<Interactions dataset ({num_users} users x {num_items} items '
                'x {num_interactions} interactions)>'
                .format(
                    num_users=self.num_users,
                    num_items=self.num_items,
                    num_interactions=len(self)
                ))

    def __len__(self):

        return len(self.user_ids)

    def _check(self):

        if self.user_ids.max() >= self.num_users:
            raise ValueError('Maximum user id greater '
                             'than declared number of users.')
        if self.item_ids.max() >= self.num_items:
            raise ValueError('Maximum item id greater '
                             'than declared number of items.')

        num_interactions = len(self.user_ids)

        for name, value in (('item IDs', self.item_ids),
                            ('ratings', self.ratings),
                            ('timestamps', self.timestamps),
                            ('weights', self.weights)):

            if value is None:
                continue

            if len(value) != num_interactions:
                raise ValueError('Invalid {} dimensions: length '
                                 'must be equal to number of interactions'
                                 .format(name))

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.user_ids
        col = self.item_ids
        data = self.ratings if self.ratings is not None else np.ones(len(self))

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()

    def to_sequence(self, max_sequence_length=10, min_sequence_length=None, step_size=None):
        """
        Transform to sequence form.

        User-item interaction pairs are sorted by their timestamps,
        and sequences of up to max_sequence_length events are arranged
        into a (zero-padded from the left) matrix with dimensions
        (num_sequences x max_sequence_length).

        Valid subsequences of users' interactions are returned. For
        example, if a user interacted with items [1, 2, 3, 4, 5], the
        returned interactions matrix at sequence length 5 and step size
        1 will be be given by:

        .. code-block:: python

           [[1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4],
            [0, 0, 1, 2, 3],
            [0, 0, 0, 1, 2],
            [0, 0, 0, 0, 1]]

        At step size 2:

        .. code-block:: python

           [[1, 2, 3, 4, 5],
            [0, 0, 1, 2, 3],
            [0, 0, 0, 0, 1]]

        Parameters
        ----------

        max_sequence_length: int, optional
            Maximum sequence length. Subsequences shorter than this
            will be left-padded with zeros.
        min_sequence_length: int, optional
            If set, only sequences with at least min_sequence_length
            non-padding elements will be returned.
        step-size: int, optional
            The returned subsequences are the effect of moving a
            a sliding window over the input. This parameter
            governs the stride of that window. Increasing it will
            result in fewer subsequences being returned.

        Returns
        -------

        sequence interactions: :class:`~SequenceInteractions`
            The resulting sequence interactions.
        """

        if self.timestamps is None:
            raise ValueError('Cannot convert to sequences, '
                             'timestamps not available.')

        if 0 in self.item_ids:
            raise ValueError('0 is used as an item id, conflicting '
                             'with the sequence padding value.')

        if step_size is None:
            step_size = max_sequence_length

        # Sort first by user id, then by timestamp
        sort_indices = np.lexsort((self.timestamps,
                                   self.user_ids))

        user_ids = self.user_ids[sort_indices]
        item_ids = self.item_ids[sort_indices]

        user_ids, indices, counts = np.unique(user_ids,
                                              return_index=True,
                                              return_counts=True)

        num_subsequences = int(np.ceil(counts / float(step_size)).sum())

        sequences = np.zeros((num_subsequences, max_sequence_length),
                             dtype=np.int32)
        sequence_users = np.empty(num_subsequences,
                                  dtype=np.int32)
        for i, (uid,
                seq) in enumerate(_generate_sequences(user_ids,
                                                      item_ids,
                                                      indices,
                                                      max_sequence_length,
                                                      step_size)):
            sequences[i][-len(seq):] = seq
            sequence_users[i] = uid

        if min_sequence_length is not None:
            long_enough = sequences[:, -min_sequence_length] != 0
            sequences = sequences[long_enough]
            sequence_users = sequence_users[long_enough]

        return (SequenceInteractions(sequences,
                                     user_ids=sequence_users,
                                     num_items=self.num_items))


class SequenceInteractions(object):
    """
    Interactions encoded as a sequence matrix.

    Parameters
    ----------

    sequences: array of np.int32 of shape (num_sequences x max_sequence_length)
        The interactions sequence matrix, as produced by
        :func:`~Interactions.to_sequence`
    num_items: int, optional
        The number of distinct items in the data

    Attributes
    ----------

    sequences: array of np.int32 of shape (num_sequences x max_sequence_length)
        The interactions sequence matrix, as produced by
        :func:`~Interactions.to_sequence`
    """

    def __init__(self,
                 sequences,
                 user_ids=None, num_items=None):

        self.sequences = sequences
        self.user_ids = user_ids
        self.max_sequence_length = sequences.shape[1]

        if num_items is None:
            self.num_items = sequences.max() + 1
        else:
            self.num_items = num_items

    def __repr__(self):

        num_sequences, sequence_length = self.sequences.shape

        return ('<Sequence interactions dataset ({num_sequences} '
                'sequences x {sequence_length} sequence length)>'
                .format(
                    num_sequences=num_sequences,
                    sequence_length=sequence_length,
                ))


def _index_or_none(array, shuffle_index):

    if array is None:
        return None
    else:
        return array[shuffle_index]


def shuffle_interactions(interactions,
                         random_state=None):
    """
    Shuffle interactions.

    Parameters
    ----------

    interactions: :class:`spotlight.interactions.Interactions`
        The interactions to shuffle.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.

    Returns
    -------

    interactions: :class:`spotlight.interactions.Interactions`
        The shuffled interactions.
    """

    if random_state is None:
        random_state = np.random.RandomState()

    shuffle_indices = np.arange(len(interactions.user_ids))
    random_state.shuffle(shuffle_indices)

    return Interactions(interactions.user_ids[shuffle_indices],
                        interactions.item_ids[shuffle_indices],
                        ratings=_index_or_none(interactions.ratings,
                                               shuffle_indices),
                        timestamps=_index_or_none(interactions.timestamps,
                                                  shuffle_indices),
                        weights=_index_or_none(interactions.weights,
                                               shuffle_indices),
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)


def random_train_test_split(interactions,
                            test_percentage=0.2,
                            random_state=None):
    """
    Randomly split interactions between training and testing.

    Parameters
    ----------

    interactions: :class:`spotlight.interactions.Interactions`
        The interactions to shuffle.
    test_percentage: float, optional
        The fraction of interactions to place in the test set.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.

    Returns
    -------

    (train, test): (:class:`spotlight.interactions.Interactions`,
                    :class:`spotlight.interactions.Interactions`)
         A tuple of (train data, test data)
    """

    interactions = shuffle_interactions(interactions,
                                        random_state=random_state)

    cutoff = int((1.0 - test_percentage) * len(interactions))

    train_idx = slice(None, cutoff)
    test_idx = slice(cutoff, None)

    train = Interactions(interactions.user_ids[train_idx],
                         interactions.item_ids[train_idx],
                         ratings=_index_or_none(interactions.ratings,
                                                train_idx),
                         timestamps=_index_or_none(interactions.timestamps,
                                                   train_idx),
                         weights=_index_or_none(interactions.weights,
                                                train_idx),
                         num_users=interactions.num_users,
                         num_items=interactions.num_items)
    test = Interactions(interactions.user_ids[test_idx],
                        interactions.item_ids[test_idx],
                        ratings=_index_or_none(interactions.ratings,
                                               test_idx),
                        timestamps=_index_or_none(interactions.timestamps,
                                                  test_idx),
                        weights=_index_or_none(interactions.weights,
                                               test_idx),
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)

    return train, test
