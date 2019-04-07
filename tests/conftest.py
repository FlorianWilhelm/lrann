#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for lrann.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""
import numpy as np
import pytest

from lrann.datasets import Interactions


@pytest.fixture
def interactions1():
    user_ids = np.array([0, 1, 1, 0, 2])
    item_ids = np.array([1, 2, 0, 2, 0])
    ratings = np.array([1, 2, 4, 4, 5])
    return Interactions(user_ids=user_ids,
                        item_ids=item_ids,
                        ratings=ratings)
