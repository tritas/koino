# coding=utf-8
from __future__ import division

import logging
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn.base import BaseEstimator
from sklearn.base import ClusterMixin
from sklearn.base import TransformerMixin

from ..plot import big_square, rect_figsize

try:
    from fastcluster import linkage
except ImportError:
    from scipy.cluster.hierarchy import linkage


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop("max_d", None)
    if max_d and "color_threshold" not in kwargs:
        kwargs["color_threshold"] = max_d
    annotate_above = kwargs.pop("annotate_above", 0)

    ddat = dendrogram(*args, **kwargs)

    if not kwargs.get("no_plot", False):
        plt.title("Hierarchical Clustering Dendrogram (truncated)")
        plt.xlabel("sample index or (cluster size)")
        plt.ylabel("distance")
        for i, d, c in zip(ddat["icoord"], ddat["dcoord"], ddat["color_list"]):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, "o", c=c)
                plt.annotate(
                    "%.3g" % y,
                    (x, y),
                    xytext=(0, -5),
                    textcoords="offset points",
                    va="top",
                    ha="center",
                )
        if max_d:
            plt.axhline(y=max_d, c="k")
    return ddat


class HierarchicalClustering(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, max_d=10, metric="euclidean", method="ward"):
        self.max_distance = max_d
        self.prediction_dist = None
        self.Z = None
        self.cluster_centers_ = None
        self.n_clusters = None
        self.cluster_counts = None
        self.metric = metric
        self.method = method

    def fit(self, X):
        self.Z = linkage(X, metric=self.metric, method=self.method)
        return self

    def predict(self, X, thr=None):
        self.prediction_dist = thr or self.max_distance
        labels = fcluster(self.Z, self.prediction_dist, criterion="distance")
        labels -= 1
        uniq, counts = np.unique(labels, return_counts=True)
        self.cluster_centers_ = np.array(
            [np.mean(X[labels == i], axis=0) for i in uniq]
        )
        self.n_clusters = len(uniq)
        self.cluster_counts = counts
        return labels

    def visualize(self, output_dir, transparent=False):
        self.plot_dendogram(output_dir, transparent=transparent)
        self.plot_knee(output_dir, transparent=transparent)

    def plot_dendogram(self, figures_dir, transparent=False):
        distance = self.prediction_dist or self.max_distance
        fig, ax = plt.subplots(1, 1, figsize=big_square)
        fancy_dendrogram(
            self.Z,
            # p=12,
            truncate_mode=None,
            show_leaf_counts=False,  # otherwise numbers in brackets are counts
            leaf_rotation=90.0,
            annotate_above=2 * distance // 3,
            max_d=distance,
            ax=ax,
        )
        dendo_path = join(
            figures_dir,
            "Dendrogram{}-{}-{}.png".format(distance, self.metric, self.method),
        )
        plt.savefig(dendo_path, transparent=transparent)
        plt.close()
        plt.clf()

    def plot_knee(self, figures_dir, n_samples=100, transparent=False):
        """Plotting the difference in distance from combining children nodes
        in the dendogram's parent node

        Parameters
        ----------
        figures_dir: str
        n_samples: int
        transparent: bool"""
        n_samples = min(n_samples, len(self.Z))
        fig, ax = plt.subplots(1, 1, figsize=rect_figsize)
        ax.plot(range(n_samples), self.Z[::-1, 2][:n_samples])
        knee = np.diff(self.Z[::-1, 2], 2)[:n_samples]
        ax.plot(range(len(knee)), knee)
        knee_path = join(figures_dir, "Knee{}.png".format(self.max_distance))
        plt.savefig(knee_path, transparent=transparent)
        plt.close()
        plt.clf()

        num_clst1 = knee.argmax() + 2
        knee[knee.argmax()] = 0
        num_clst2 = knee.argmax() + 2
        logging.info("[HAC] Knees: #1={}, #2={}".format(num_clst1, num_clst2))

    def fit_predict(self, X, thr=None):
        return self.fit(X).predict(X, thr)
