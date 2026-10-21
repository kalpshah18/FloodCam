"""Tests for the seeding utility (src/floodcam/seed.py)."""

import random

import numpy as np
from floodcam.seed import seed_everything


def test_numpy_reproducible():
    seed_everything(42)
    a = np.random.rand(10)
    seed_everything(42)
    b = np.random.rand(10)
    np.testing.assert_array_equal(a, b)


def test_random_reproducible():
    seed_everything(42)
    a = [random.random() for _ in range(10)]
    seed_everything(42)
    b = [random.random() for _ in range(10)]
    assert a == b


def test_different_seeds_produce_different_values():
    seed_everything(1)
    a = np.random.rand(10)
    seed_everything(2)
    b = np.random.rand(10)
    assert not np.array_equal(a, b)


def test_seed_idempotent():
    seed_everything(99)
    seed_everything(99)
    a = np.random.rand(5)
    seed_everything(99)
    seed_everything(99)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)
