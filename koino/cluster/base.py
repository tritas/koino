# coding=utf-8
from __future__ import division

import logging
from itertools import permutations
from os.path import join
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from joblib import Memory
from koino.plot.base import hist, matshow_colorbar
from .stability import ClusteringAnalysis
from scipy.sparse import csr_matrix, issparse, spmatrix
from sklearn import cluster
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralCoclustering
from sklearn.neighbors import kneighbors_graph

from .hierarchical import HierarchicalClustering
from ..plot.clusters import plot_cluster_assignments

clustering_algorithms = [
    "KMeans",
    "AffinityPropagation",
    "MeanShift",
    "SpectralClustering",
    "Ward",
    "AgglomerativeClustering",
    "AgglomerativeCosine",
    "DBSCAN",
    "Birch",
]
logger = logging.getLogger(__name__)


random_state_typ = Optional[Union[int, np.random.RandomState]]


def default_kmeans(
    n_clusters: int,
    n_samples: Optional[int] = None,
    verbose: int = 0,
    random_state: random_state_typ = None,
) -> KMeans:
    """ Sensible defaults for clustering that is a tiny bit more robust.
    Full-batch KMeans is unbearably slow for large datasets.
    Can init more than 100 times, how much time do you have?
    """
    if n_samples and n_samples > int(1e5):
        instance = MiniBatchKMeans(
            max_no_improvement=100,
            batch_size=1000,
            verbose=verbose,
            max_iter=400,
            n_init=100,
            n_clusters=n_clusters,
            random_state=random_state,
        )
    else:
        instance = KMeans(
            n_clusters=n_clusters,
            n_jobs=-1,
            max_iter=400,
            n_init=100,
            random_state=random_state,
            precompute_distances=True,
            verbose=verbose,
        )
    return instance


def clustering(
    X: Union[pd.DataFrame, np.ndarray],
    algorithm: str,
    n_clusters: int = 10,
    verbose: int = 0,
    random_state: random_state_typ = None,
) -> np.ndarray:
    """Compute cluster assignments for given array and clustering algorithm."""
    model = None
    n_samples = X.shape[0]
    # Choose algo
    if algorithm == "KMeans":
        model = default_kmeans(n_clusters, n_samples=n_samples)

    elif algorithm == "Birch":
        model = cluster.Birch(n_clusters=n_clusters)

    elif algorithm == "DBSCAN":
        model = cluster.DBSCAN(n_jobs=-1)

    elif algorithm == "AffinityPropagation":
        model = cluster.AffinityPropagation(verbose=verbose)

    elif algorithm == "MeanShift":
        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(
            X, quantile=0.3, random_state=random_state, n_jobs=-1
        )
        logger.debug("[MeanShift] Bandwith={}".format(bandwidth))
        model = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)

    elif algorithm == "SpectralClustering":
        model = cluster.SpectralClustering(
            n_clusters=n_clusters,
            eigen_solver="arpack",
            affinity="nearest_neighbors",
            n_init=100,
            n_jobs=-1,
        )

    elif algorithm in ("Ward", "AgglomerativeClustering", "AgglomerativeCosine"):
        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            X, n_neighbors=10, include_self=False, n_jobs=-1
        )

        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)
        if algorithm == "Ward":
            model = cluster.AgglomerativeClustering(
                n_clusters=n_clusters, linkage="ward", connectivity=connectivity
            )
        elif algorithm == "AgglomerativeClustering":
            model = cluster.AgglomerativeClustering(
                linkage="average",
                affinity="euclidean",
                n_clusters=n_clusters,
                connectivity=connectivity,
            )
        elif algorithm == "AgglomerativeCosine":
            model = cluster.AgglomerativeClustering(
                n_clusters=n_clusters, affinity="cosine", linkage="average"
            )

    model.fit(X)

    if hasattr(model, "labels_"):
        y_pred = model.labels_.astype(np.int)
    else:
        y_pred = model.predict(X)

    return y_pred


def compare_clustering(
    X: np.ndarray,
    X_tsne: np.ndarray,
    n_clusters: int,
    figures_dir: str,
    verbose: int,
    transparent: int,
) -> Dict[str, np.ndarray]:
    labels = {}
    for algo in clustering_algorithms:
        logger.info("[Clustering] Algo: {}".format(algo))
        assignments = clustering(X, algo, n_clusters, verbose)
        plot_cluster_assignments(
            X_tsne,
            assignments,
            n_clusters,
            figures_dir,
            transparent=transparent,
            title=algo,
        )
        labels[algo] = assignments
    return labels


def init_coclustering(
    rows: np.ndarray,
    cols: np.ndarray,
    output_dir: Optional[str] = None,
    row_label: Optional[str] = "Socio-demo",
    col_label: Optional[str] = "Diet",
    transparent: bool = False,
) -> csr_matrix:
    dat = np.ones_like(rows, dtype=np.float32)
    X_sp = csr_matrix((dat, (rows, cols)), dtype=np.float32)
    if output_dir:
        hist_path = join(output_dir, "Co-clustering assignments histogram")
        hist(X_sp.data, hist_path, xlabel=row_label, ylabel=col_label)
        cooc_path = join(output_dir, "Cooc values histogram")
        cooc_title = "Co-clustering original matrix"
        matshow_colorbar(
            X_sp.A,
            cooc_path,
            cooc_title,
            xlabel=row_label,
            ylabel=col_label,
            transparent=transparent,
        )
    return X_sp


def spectral_coclustering(
    X: Union[np.ndarray, spmatrix],
    n_clusters: int,
    output_dir: Optional[str] = None,
    row_label: Optional[str] = "Socio-demo",
    col_label: Optional[str] = "Diet",
    transparent=False,
    random_state: random_state_typ = None,
) -> Union[np.ndarray, spmatrix]:
    """ Run spectral co-clustering on a sparse or dense matrix and
    visualize the result.

    Parameters
    ----------
    X: numpy array, scipy sparse matrix of shape [n, m]
    n_clusters: int
    output_dir: str, path
    row_label: str
    col_label: str
    transparent: bool
    random_state: int, np.random.RandomState or None

    Returns
    -------
    X_perm: numpy array, scipy sparse matrix of shape [n, m]
    """
    model = SpectralCoclustering(
        n_clusters=n_clusters, random_state=random_state, n_jobs=-1
    )
    model.fit(X)
    X_perm = X[np.argsort(model.row_labels_)]
    X_perm = X_perm[:, np.argsort(model.column_labels_)]

    fit_data_dense = X_perm.A if issparse(X_perm) else X_perm
    if output_dir:
        figpath = join(output_dir, "spectral_coclst_{}.png".format(n_clusters))
        matshow_colorbar(
            fit_data_dense,
            figpath,
            "Rearranged clusters",
            xlabel=row_label,
            ylabel=col_label,
            transparent=transparent,
        )
    # XXX: This just takes ages, do not use blindly
    # score = consensus_score(model.biclusters_, X.nonzero())
    # logger.info('[Coclustering] Consensus score={}'.format(score))
    return X_perm


def vector_quantization(
    X: np.ndarray,
    n_clusters: Optional[int] = None,
    clusters_span: Optional[Union[List[int], Tuple[int]]] = None,
    hac_dist: Optional[float] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, int, ClusterMixin]:
    """ Solve a vector quantization problem and optionally visualize
    clustering assignments.

    Parameters
    ----------
    X: numpy array of shape [n_samples, n_features]
    n_clusters: int (optional)
    clusters_span: range (optional)
    hac_dist:

    Returns
    -------
    assignments: numpy array of shape [n_samples,]
    centroids: numpy array of shape [n_clusters, n_features]
    n_clusters: int
    clst: object
    """
    if clusters_span:
        # Check that clusters' span is correct
        if isinstance(clusters_span, (list, tuple)):
            clusters_span = range(*clusters_span)
        assert clusters_span.start >= 2, clusters_span.start
        assert len(clusters_span) >= 2, clusters_span
        if n_clusters:
            assert clusters_span.start <= n_clusters <= clusters_span.stop
        logger.info("[VQ] Clusters spanning {}".format(clusters_span))
        # Find optimal number of clusters and predict labels
        clst = ClusteringAnalysis(
            clustering_model=default_kmeans, clusters_span=clusters_span, **kwargs
        )
    elif n_clusters:
        logger.info("[VQ] Quantizing with {} centroids".format(n_clusters))
        clst = default_kmeans(n_clusters)
    elif hac_dist:
        logger.info("[HAC] Quantizing at distance {}".format(hac_dist))
        clst = HierarchicalClustering(hac_dist)
    else:
        raise ValueError("No. clusters, clusters span or HAC dist expectd")

    assignments = clst.fit_predict(X)
    centroids = clst.cluster_centers_
    n_clusters = clst.n_clusters

    return assignments, centroids, n_clusters, clst


def run_hdbscan(X_df, X_tsne, output_dir, transparent):
    """Cluster using density estimation

     Parameters
     ----------
     X_df: DataFrame
     X_tsne: array-like, [n_samples, 2]
     output_dir: str, path
     transparent: bool

     Returns
     -------
     clusterer: HDBSCAN object
     assignments: numpy array of shape [n_samples,]

     """
    from hdbscan import HDBSCAN

    clusterer = HDBSCAN(
        core_dist_n_jobs=-1,
        cluster_selection_method="eom",  # 'leaf',
        approx_min_span_tree=False,
        min_cluster_size=100,
        min_samples=1,
        leaf_size=100,
        gen_min_span_tree=True,
        # alpha=10.,
        memory=Memory(cachedir=None, verbose=0),
    )

    assignments = clusterer.fit_predict(X_df)
    centroid_labels, counts = np.unique(assignments, return_counts=True)
    n_clusters = len(centroid_labels)
    assignments[assignments == -1] = n_clusters - 1

    logger.info("[HDBSCAN] Found {} clusters".format(n_clusters))
    logger.info("[HDBSCAN] Cluster assignments:\n{}".format(counts))
    logger.info(
        "[HDBSCAN] Cluster persistence:\n{}".format(clusterer.cluster_persistence_)
    )
    return assignments, clusterer.exemplars_, n_clusters, clusterer


def visualize_hdbscan(
    clusterer, X_projected, assignments, n_clusters, output_dir, transparent
):
    """ Visualize HDBSCAN results
    Parameters
    ----------
    clusterer: object
    X_projected: array - like, [n_samples, 2]
    assignments
    n_clusters
    output_dir: str, path
    transparent
    """
    probas_fp = join(output_dir, "HDBSCAN_sample_probas.png")
    outliers_fp = join(output_dir, "HDBSCAN_outliers.png")
    hist(clusterer.probabilities_, probas_fp)
    hist(clusterer.outlier_scores_, outliers_fp)
    plot_cluster_assignments(
        X_projected,
        assignments,
        "HDBSCAN assignments",
        n_clusters,
        output_dir,
        transparent,
    )


def meila_distance(clustering1, clustering2, num_clusters):

    n_samples = len(clustering1)

    clustering_1 = np.zeros((n_samples, num_clusters))
    clustering_2 = np.zeros((n_samples, num_clusters))

    for x in range(0, n_samples):
        clustering_1[x, clustering1[x]] += 1
        clustering_2[x, clustering2[x]] += 1

    confusion_matrix = np.dot(np.transpose(clustering_1), clustering_2)

    max_confusion = 0

    for perm in permutations(range(0, num_clusters)):
        confusion = 0
        for i in range(0, num_clusters):
            confusion += confusion_matrix[i, perm[i]]

        if max_confusion < confusion:
            max_confusion = confusion

    distance = 1 - (max_confusion / n_samples)

    return distance
