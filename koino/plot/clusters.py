# coding=utf-8
import logging
import traceback
from os import makedirs
from os.path import exists, join
from textwrap import fill

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from koino.plot import big_square, default_alpha
from matplotlib import cm

from ..utils.base import jaccard


def plot_silhouette(
    X, figure_fp, n_clusters, silhouette_values, cluster_labels, silhouette_avg
):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 10))

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but here  all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    y_lower = 10
    for k in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = np.sort(silhouette_values[cluster_labels == k])

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(k) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=default_alpha,
        )

        # Label the silhouette plots with their cluster numbers at the
        # middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(k))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # Construct cluster
    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    # colors = y
    ax2.scatter(X[:, 0], X[:, 1], marker=".", s=20, lw=0, alpha=default_alpha, c=colors)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        ("Silhouette analysis for KMeans " "with n_clusters = %d" % n_clusters),
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(figure_fp)
    plt.close()
    plt.clf()


def plot_cluster_assignments(
    X, y, n_clusters, figures_dir, transparent=False, cluster_names=None, title=""
):
    """Clustering assignments scatter plot
    Notes
    -----
    Can use mean or median to fix cluster centroid coordinates."""
    if cluster_names is None:
        cluster_names = ["Cluster {}".format(i + 1) for i in range(n_clusters)]

    # We first reorder the data points according to the centroids labels
    X = np.vstack([X[y == i] for i in range(n_clusters)])
    y = np.hstack([y[y == i] for i in range(n_clusters)])

    # Choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", n_clusters))

    fig, ax = plt.subplots(figsize=big_square)
    # for i in range(n_clusters):
    #     mask = y == i
    #     ax.scatter(X[mask, 0], X[mask, 1], lw=0, s=20, c=palette[i],
    # label=cluster_names[i])
    ax.set_title(title)
    ax.scatter(X[:, 0], X[:, 1], lw=0, s=20, c=palette[y.astype(np.int)])
    ax.axis("off")

    # Add the labels for each cluster.
    for i in range(n_clusters):
        # Position of each label.
        samples = np.atleast_2d(X[y == i, :2])
        if not len(samples):
            logging.warning(
                "Probably singular cluster {} (shape:{})".format(i + 1, X[y == i].shape)
            )
            continue
        xtext, ytext = np.median(samples, axis=0)
        name = fill(cluster_names[i], width=20)
        assert np.isfinite(xtext)
        assert np.isfinite(ytext)
        txt = ax.text(xtext, ytext, name, fontsize=20, wrap=True, ha="left")
        txt.set_path_effects(
            [PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()]
        )
    # plt.legend()
    figure_fp = join(figures_dir, "Clustered {}.png".format(title))
    fig.tight_layout()
    try:
        fig.savefig(figure_fp, transparent=transparent)
    except ValueError:
        logging.warning(traceback.format_exc())
    finally:
        plt.close()
        plt.clf()


def overlap_jaccard(
    indx,
    y_a,
    y_b,
    names_a,
    names_b,
    n_a=None,
    n_b=None,
    figsize=None,
    output_dir=None,
    alabel="socio-demographic",
    blabel="purchases",
    transparent=False,
):
    """Compute and plot contingency tables based on set intersection and
    jaccard score.

    # TODO: Normaliser par len(sd_set) ou len(diet_set) ?
    """
    if not (n_a or n_b) or not output_dir:
        return
    elif output_dir and not exists(output_dir):
        makedirs(output_dir)
    else:
        assert n_a and n_b

    assert len(indx) == len(y_a) == len(y_b)
    assert len(names_a) == n_a
    assert len(names_b) == n_b

    a_sets = [set(indx[y_a == i]) for i in range(n_a)]
    b_sets = [set(indx[y_b == i]) for i in range(n_b)]

    inter_sets = np.asarray(
        [[len(set_a & set_t) for set_a in a_sets] for set_t in b_sets], dtype=np.int_
    )

    fig, ax = plt.subplots(figsize=figsize)
    plt.title("Overlap between {} and {} clusters".format(alabel, blabel))
    sns.heatmap(
        inter_sets,
        annot=True,
        fmt="6.0f",
        ax=ax,
        square=True,
        xticklabels=names_a,
        yticklabels=names_b,
    )
    plt.tight_layout()
    inter_path = join(output_dir, "Clusters Intersection.png")
    plt.savefig(inter_path, transparent=transparent)
    plt.close()
    plt.clf()

    jac_arr = np.asarray(
        [[jaccard(set_a, set_b) for set_a in a_sets] for set_b in b_sets],
        dtype=np.float_,
    )

    fig, ax = plt.subplots(figsize=figsize)
    plt.title("Jaccard scores between {} and {} clusters".format(alabel, blabel))
    sns.heatmap(
        jac_arr,
        annot=True,
        fmt=".3f",
        ax=ax,
        square=True,
        xticklabels=names_a,
        yticklabels=names_b,
    )
    plt.tight_layout()
    jaccard_path = join(output_dir, "Clusters Jaccard.png")
    plt.savefig(jaccard_path, transparent=transparent)
    plt.close()
    plt.clf()
