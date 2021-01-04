# coding=utf-8
from tempfile import mkdtemp

import numpy as np
import scipy.sparse as sp
from sklearn.utils import shuffle

from kantar.cluster.base import spectral_coclustering

tmpdir = mkdtemp()


def test_spectral():
    A = sp.coo_matrix(
        [
            [3, 3, 2, 2, 1, 1],
            [2, 2, 3, 3, 1, 1],
            [1, 1, 2, 2, 3, 3],
            [3, 3, 2, 2, 1, 1],
            [2, 2, 3, 3, 1, 1],
            [1, 1, 2, 2, 3, 3],
        ]
    )
    B, score = spectral_coclustering(A.row, A.col, 3, tmpdir)
    print()
    print(B.A)
    print(score)
    assert score <= 0.15


def test_spectral_binary():
    A = sp.coo_matrix(
        [
            [0, 0, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0],
        ]
    )
    B, score = spectral_coclustering(A.row, A.col, 3, tmpdir)
    print()
    print(B.A)
    print(score)
    assert score == 0.


def test_spectral_big():
    # TODO: Add more more entries per row/column
    n = 2000
    rnd = np.random.rand(n, n) < 1 / n
    sp_rnd = sp.coo_matrix(rnd, dtype=np.float64)
    eye = sp.eye(n, dtype=np.float64, format="coo")
    A = shuffle(eye + sp_rnd)
    B, score = spectral_coclustering(A.row, A.col, 30, tmpdir)
    print(score)
