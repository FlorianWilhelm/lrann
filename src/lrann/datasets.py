# -*- coding: utf-8 -*-
"""
Different data sets

Note: Some parts copied over from Spotlight
"""
import os
import shutil
import logging
from zipfile import ZipFile
from collections import namedtuple, defaultdict

import numpy as np
import scipy.sparse as sparse
import pandas as pd
import requests
from tqdm import tqdm

_logger = logging.getLogger(__name__)

Resource = namedtuple('Resource', ['url', 'path', 'interactions', 'read_csv_args'])

# all available datasets
MOVIELENS_20M = Resource(
    path='ml-20m/raw.zip',
    interactions='ratings.csv',
    read_csv_args={'names': ['user_id', 'item_id', 'rating', 'timestamp'],
                   'header': 0},
    url='http://files.grouplens.org/datasets/movielens/ml-20m.zip')
MOVIELENS_100K_OLD = Resource(
    path='ml100k-old/raw.zip',
    interactions='u.data',
    read_csv_args={'names': ['user_id', 'item_id', 'rating', 'timestamp'],
                   'header': None,
                   'sep': '\t'},
    url='http://files.grouplens.org/datasets/movielens/ml-100k.zip')
MOVIELENS_100K = Resource(
    path='ml-latest-small/raw.zip',
    interactions='ratings.csv',
    read_csv_args={'names': ['user_id', 'item_id', 'rating', 'timestamp'],
                   'header': 0},
    url='http://files.grouplens.org/datasets/movielens/ml-latest-small.zip')

DATA_DIR = os.path.join(os.path.expanduser('~'), '.lrann')


def compact(iterable):
    """Applies Compacter to get consecutive elements, e.g.
    [1, 2, 5, 2] or ['a', 'b', 'e', 'b'] become
    a compact, consecutive integer representation [0, 1, 2, 1]
    """
    mapper = defaultdict()
    mapper.default_factory = mapper.__len__
    return np.array([mapper[elem] for elem in iterable], dtype=np.int32)


class Loader(object):
    def __init__(self, data_dir=DATA_DIR, show_progress=True):
        self.data_dir = data_dir
        self.show_progress = show_progress

    def get_data(self, resource, download_if_missing=True):
        dest_path = os.path.join(os.path.abspath(self.data_dir), resource.path)
        dir_path = os.path.dirname(dest_path)
        create_data_dir(dir_path)

        if not os.path.isfile(dest_path):
            if download_if_missing:
                download(resource.url, dest_path, self.show_progress)
            else:
                raise IOError('Dataset missing.')
        if count_files(dir_path) == 1:
            unzip(dest_path, dir_path)

        return dir_path

    def load_movielens(self, variant='100k'):
        variants = {'100k': MOVIELENS_100K,
                    '100k-old': MOVIELENS_100K_OLD,
                    '20m': MOVIELENS_20M}
        resource = variants[variant]
        dir_path = self.get_data(resource)
        ratings = os.path.join(dir_path, resource.interactions)
        df = pd.read_csv(ratings, **resource.read_csv_args)
        user_ids = df.user_id.values
        item_ids = df.item_id.values
        ratings = df.rating.values
        user_ids = compact(user_ids)
        item_ids = compact(item_ids)
        interactions = Interactions(user_ids=user_ids,
                                    item_ids=item_ids,
                                    ratings=ratings)
        return interactions


def download(url, dest_path, show_progress=True, chunk_size=1024):
    req = requests.get(url, stream=True)
    req.raise_for_status()

    bytestream = req.iter_content(chunk_size=chunk_size)
    if show_progress:
        file_size = int(req.headers['Content-Length'])
        n_bars = file_size // chunk_size

        bytestream = tqdm(bytestream,
                          unit='KB',
                          total=n_bars,
                          ascii=True,
                          desc=dest_path)

    with open(dest_path, 'wb') as fd:
        for chunk in bytestream:
            fd.write(chunk)


def unzip(src_path, dest_dir):
    with ZipFile(src_path) as zip_file:
        for obj in zip_file.filelist:
            if obj.is_dir():
                continue
            source = zip_file.open(obj.filename)
            filename = os.path.basename(obj.filename)
            with open(os.path.join(dest_dir, filename), "wb") as target:
                shutil.copyfileobj(source, target)


def create_data_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def count_files(path):
    return len([name for name in os.listdir(path)
                if os.path.isfile(os.path.join(path, name))])


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
    n_users: int, optional
        Number of distinct users in the dataset.
    n_items: int, optional
        Number of distinct items in the dataset.
    """

    def __init__(self, user_ids, item_ids,
                 ratings=None,
                 timestamps=None,
                 weights=None,
                 n_users=None,
                 n_items=None):

        self.n_users = n_users or int(user_ids.max() + 1)
        self.n_items = n_items or int(item_ids.max() + 1)

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.timestamps = timestamps
        self.weights = weights

        self._check()

    def __repr__(self):

        return ('<Interactions dataset ({n_users} users x {n_items} items '
                'x {n_interactions} interactions)>'
                .format(
                    n_users=self.n_users,
                    n_items=self.n_items,
                    n_interactions=len(self)
                ))

    def __len__(self):
        return len(self.user_ids)

    def _check(self):
        if self.user_ids.max() >= self.n_users:
            raise ValueError('Maximum user id greater '
                             'than declared number of users.')
        if self.item_ids.max() >= self.n_items:
            raise ValueError('Maximum item id greater '
                             'than declared number of items.')

        n_interactions = len(self.user_ids)

        for name, value in (('item IDs', self.item_ids),
                            ('ratings', self.ratings),
                            ('timestamps', self.timestamps),
                            ('weights', self.weights)):

            if value is None:
                continue

            if len(value) != n_interactions:
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

        return sparse.coo_matrix((data, (row, col)),
                                 shape=(self.n_users, self.n_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()

    def topandas(self):
        """
        Transform to Pandas DataFrame
        """
        data = np.column_stack([self.user_ids, self.item_ids, self.ratings])
        return pd.DataFrame(data=data,
                            columns=['user_id', 'item_id', 'rating'],
                            dtype=np.int32)


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
                        n_users=interactions.n_users,
                        n_items=interactions.n_items)


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
                         n_users=interactions.n_users,
                         n_items=interactions.n_items)
    test = Interactions(interactions.user_ids[test_idx],
                        interactions.item_ids[test_idx],
                        ratings=_index_or_none(interactions.ratings,
                                               test_idx),
                        timestamps=_index_or_none(interactions.timestamps,
                                                  test_idx),
                        weights=_index_or_none(interactions.weights,
                                               test_idx),
                        n_users=interactions.n_users,
                        n_items=interactions.n_items)

    return train, test
