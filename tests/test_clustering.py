# coding=utf-8
import logging
from tempfile import mkdtemp

import pytest
from numpy import unique
from sklearn.datasets import make_blobs

from koino.cluster.base import ClusteringAnalysis
from koino.cluster.base import vector_quantization
from koino.cluster.hierarchical import HierarchicalClustering

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG
)

tmpdir = mkdtemp()
logging.info(tmpdir)

n_samples = 1200
seed = 42
X, y = make_blobs(n_samples=n_samples, n_features=10, centers=13, random_state=seed)

# transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
# X_aniso = np.dot(X, transformation)

# X_varied, y_varied = make_blobs(
#     n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=seed)


def test_hac():
    hac = HierarchicalClustering()
    labels = hac.fit_predict(X)
    assert 2 < len(unique(labels)) < 100


@pytest.mark.skip(reason="no assertion at the end")
def test_clustering_analysis(span=range(10, 15), n_bootstrap=20, noise_std=0.5):
    clst = ClusteringAnalysis(
        clusters_span=span,
        bootstrap=n_bootstrap,
        subsamples=n_samples // n_bootstrap,
        samples=n_samples // n_bootstrap,
        noise_scale=noise_std,
    )
    clst.fit(X)
    k_opt = clst.cluster_centers_.shape[1]
    assert span.start <= k_opt <= span.stop


def test_no_vq():
    labels, centroids, _, _ = vector_quantization(X, X, tmpdir)
    assert labels is None
    assert centroids is None


@pytest.mark.parametrize(
    "n_clusters,span",
    [(10, None), (None, range(2, 4)), (None, range(3, 7, 2))],
)
def test_vq(n_clusters, span):
    labels, centroids, _, _ = vector_quantization(X, X, tmpdir, n_clusters, span, False)
    assert len(labels) == len(X)
    assert len(centroids) == n_clusters
