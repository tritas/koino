import logging
from collections import defaultdict
from itertools import combinations
from os.path import join
from time import time
from typing import List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.base import ClusterMixin
from sklearn.base import TransformerMixin
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_samples

from .base import default_kmeans
from ..plot.clusters import plot_silhouette

logger = logging.getLogger(__name__)


def compute_stability_metrics(
    clst: ClusterMixin,
    X: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    bootstrap: int,
    noise_scale: float,
) -> List[Tuple[str, str, Tuple[float, float]]]:
    """We evaluate the adjusted mutual information score between a ground truth
    clustering and perturbed versions of the dataset.
    We generate perturbation by adding isotropic gaussian noise, subsampling
    with and without replacement.

    Parameters
    ----------
    clst: object
        Clustering algorithm to use, must implement `predict` method.
    X: numpy array of [n_samples, n_features]
    labels: numpy array of [n_samples,]
        Ground truth clustering assignments
    n_clusters: int
        Hypothesised number of clusters in the data.
    bootstrap: int
    noise_scale: float

    Returns
    -------
    results: list<Metric, Comparison Typ, Bootstrap scores 1st and 2nd moment>
        Structure that holds the metrics' results for two different
        types of comparison of the (generated) perturbed dataset.
    """
    logger.debug(
        "[Stability metrics] clusters={}, bootstrap={}, noise={}".format(
            n_clusters, bootstrap, noise_scale
        )
    )
    metric_funcs = {
        "Adjusted Rand Score": adjusted_rand_score,
        "Adjusted Mutual Information": adjusted_mutual_info_score,
        "Normalized Mutual Information": normalized_mutual_info_score,
    }
    noisy_pairwise = "Noisy-to-Noisy, scale={:.1f}".format(noise_scale)
    noisy_to_orig = "Noisy-to-Original, scale={:.1f}".format(noise_scale)
    results = list()
    # Predict labels for perturbed version of the dataset
    noisy_labels = []
    for j in range(bootstrap):
        X_noisy = X + np.random.randn(*X.shape) * noise_scale
        pred_labels = clst.predict(X_noisy)
        noisy_labels.append(pred_labels)
    # Build consensus indices
    for metric, func in metric_funcs.items():
        # Comparing perturbed versions to original dataset
        scores = []
        for j in range(bootstrap):
            score = func(labels, noisy_labels[j])
            scores.append(score)
        scores = np.array(scores)
        results.append((metric, noisy_to_orig, (scores.mean(), scores.std())))
        # Comparing all perturbed versions between themselves
        scores = []
        for j, k in combinations(range(bootstrap), 2):
            score = func(noisy_labels[j], noisy_labels[k])
            scores.append(score)
        scores = np.array(scores)
        results.append((metric, noisy_pairwise, (scores.mean(), scores.std())))
    return results


class ClusteringAnalysis(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(
        self,
        clustering_model: ClusterMixin,
        clusters_span: Tuple[int],
        bootstrap: int = 10,
        noise_scale: float = 0.1,
    ):
        """Estimator that seeks the optimal number of clusters for a given
        problem solved with kmeans++ and multiple restarts.

        Parameters
        ----------
        clusters_span: tuple
        bootstrap: int
        noise_scale: float
        """
        self.clustering_model = clustering_model
        self.n_clusters = None
        self.span = clusters_span
        self.clst = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.labels = dict()

        # Sample sizes
        self.bootstrap = bootstrap
        self.noise_scale = noise_scale

        # Metrics
        n_iters = len(clusters_span)
        self.inertia = np.zeros(n_iters)
        self.calinski_harabaz_scores = np.zeros(n_iters)
        # Mean, Std inertia metrics for each run according to uniform distrib.
        self.unif_inertia = np.zeros((n_iters, 2))
        # Mean, Std silhouette metrics for each run
        self.silhouette_stats = np.zeros((n_iters, 2))
        self.silhouette_scores = []
        self.stability_metrics = defaultdict(lambda: defaultdict(list))

    def fit(self, X):
        best_silhouette_score = 0.0
        best_clst = None
        k_opt = -1

        if len(X) > int(1e5):
            sample_is = np.random.choice(len(X), 10000, replace=False)
        else:
            sample_is = slice(None)

        for i, n_clusters in enumerate(self.span):
            # Cluster data
            clst = self.clustering_model(n_clusters, n_samples=len(X))
            t0 = time()
            labels = clst.fit_predict(X)
            t1 = time()
            self.labels[n_clusters] = labels
            self.inertia[i] = clst.inertia_
            self.calinski_harabaz_scores[i] = calinski_harabaz_score(X, labels)
            # Compute the silhouette scores for each sample
            X_silhouette = silhouette_samples(X[sample_is], labels[sample_is])
            silhouette_avg = np.mean(X_silhouette)
            silhouette_std = np.std(X_silhouette)
            self.silhouette_stats[i, :] = silhouette_avg, silhouette_std
            self.silhouette_scores.append(X_silhouette)
            # Evaluate clustering stability
            iter_scores = compute_stability_metrics(
                clst,
                X[sample_is],
                labels[sample_is],
                n_clusters,
                self.bootstrap,
                self.noise_scale,
            )
            for metric, comparison, value_tup in iter_scores:
                self.stability_metrics[metric][comparison].append(value_tup)
            logger.info(
                "[KMeans k={:3}] Iters={:3} Time={:.3f}s "
                "Silhouette avg.={:.3f} +/- {:.3f}".format(
                    n_clusters, clst.n_iter_, t1 - t0, silhouette_avg, silhouette_std
                )
            )
            logger.info(iter_scores)

            if silhouette_avg > best_silhouette_score:
                logger.info("[KMeans k={:2}] New best".format(n_clusters))
                best_silhouette_score = silhouette_avg
                best_clst = clst
                k_opt = n_clusters

        self.clst = best_clst
        self.n_clusters = k_opt
        self.cluster_centers_ = best_clst.cluster_centers_
        self.labels_ = self.labels[k_opt]

        return self

    def transform(self, X):
        return self.clst.transform(X)

    def predict(self, X):
        return self.labels_

    def visualize(
        self,
        figures_dir: str,
        figsize: Tuple[Union[int, float]] = (12.8, 9.6),
        transparent: bool = False,
        embeddings: Optional[np.ndarray] = None,
        alpha: Optional[float] = 0.5,
    ):
        """Visualize the clustering metrics.

        Parameters
        ----------
        figures_dir: str
        figsize: tuple
        transparent: bool
        embeddings: array-like of shape [n_samples, embedding_dim]
        alpha: float
          Figure transparency.
        """
        begin, end = self.span.start, self.span.stop
        n_clusters_lst = list(self.span)
        log_inertia = np.log(self.inertia)
        # Plot dataset distortion
        plt.figure(figsize=figsize)
        plt.plot(n_clusters_lst, log_inertia, label="Distortion")
        plt.title("Evolution of log inertia with the number of clusters")
        plt.xticks(n_clusters_lst)
        plt.xlabel("Clusters")
        plt.ylabel("Log-Inertia")
        plt.legend()
        plt.tight_layout()
        log_inertia_fn = "log_inertia_{}_{}.png".format(begin, end)
        plt.savefig(join(figures_dir, log_inertia_fn), transparent=transparent)
        plt.clf()
        plt.close()

        plt.figure(figsize=figsize)
        plt.plot(n_clusters_lst, self.inertia, label="Distortion")
        plt.title("Evolution of inertia with the number of clusters")
        plt.xlabel("Clusters")
        plt.ylabel("Inertia")
        plt.xticks(n_clusters_lst)
        plt.legend()
        inertia_fn = "inertia_{}_{}.png".format(begin, end)
        plt.tight_layout()
        plt.savefig(join(figures_dir, inertia_fn), transparent=transparent)
        plt.clf()
        plt.close()

        # --- Plot average silhouette
        silhouette_fn = "silhouette_{}_{}.png".format(begin, end)
        # In case we skipped some
        plt.figure(figsize=figsize)
        plt.plot(n_clusters_lst, self.silhouette_stats[:, 0])
        plt.fill_between(
            n_clusters_lst,
            self.silhouette_stats[:, 0] - self.silhouette_stats[:, 1],
            self.silhouette_stats[:, 0] + self.silhouette_stats[:, 1],
            alpha=alpha,
        )
        plt.xlabel("Clusters")
        plt.ylabel("Silhouette score avg.")
        plt.xticks(n_clusters_lst)
        plt.title("Silhouette avg. for K=({}, {})".format(begin, end))
        plt.savefig(join(figures_dir, silhouette_fn), transparent=transparent)
        plt.close()
        plt.clf()

        if embeddings is not None:
            # --- Plot silhouettes
            for i, n_clusters in enumerate(self.span):
                figure_filename = "Silhouette_{}.png".format(n_clusters)
                figure_path = join(figures_dir, figure_filename)
                plot_silhouette(
                    embeddings,
                    figure_path,
                    n_clusters,
                    self.silhouette_scores[i],
                    self.labels[n_clusters],
                    self.silhouette_stats[i, 0],
                )

        # --- Plot stability metrics
        for metric_name, results_dict in self.stability_metrics.items():
            plt.figure(figsize=figsize)
            plt.xlabel("Clusters")
            plt.ylabel("{} (a.u.)".format(metric_name))
            for comparison_typ, moment_tuples in sorted(results_dict.items()):
                mean, std = np.array(moment_tuples).T
                plt.plot(n_clusters_lst, mean, label=comparison_typ)
                plt.fill_between(n_clusters_lst, mean - std, mean + std, alpha=alpha)
            plt.xticks(n_clusters_lst)
            plt.title("Evolution of {}".format(metric_name))
            plt.legend()
            metric_path = "{} {}-{}.png".format(metric_name, begin, end)
            metric_path = join(figures_dir, metric_path)
            plt.savefig(metric_path, transparent=transparent)
            plt.close()
            plt.clf()

        # Calinski-Harabaz Index (higher is better)
        ch_path = "calinsk_harabaz_{}_{}.png".format(begin, end)
        plt.figure(figsize=figsize)
        plt.xlabel("Clusters")
        plt.ylabel("Calinski-Harabaz score")
        plt.plot(n_clusters_lst, self.calinski_harabaz_scores)
        plt.xticks(n_clusters_lst)
        plt.title("Evolution of Calinski-Harabaz score")
        plt.savefig(join(figures_dir, ch_path), transparent=transparent)
        plt.clf()
        plt.close()

    plt.close("all")
