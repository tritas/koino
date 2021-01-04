# coding=utf-8
import logging
from os.path import join
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from seaborn import boxplot
from seaborn import heatmap
from sklearn.preprocessing import normalize
from statsmodels.stats.weightstats import DescrStatsW

from .tests import vtests
from .tests import ttests
from ..plot.base import make_annotated_pie
from ..utils.base import remove_missing
from ..utils.base import to_percent
from ..utils.base import weighted_value_counts

logger = logging.getLogger(__name__)


def check_clustering(
    assignments: np.ndarray, names: Optional[List[str]] = None
) -> None:
    """Check which clusters are singular and report them."""
    _, counts = np.unique(assignments, return_counts=True)
    singular = counts < 2
    if np.any(singular):
        indx = np.argwhere(singular)
        malformed = np.asarray(names)[indx] if names is not None else indx
        logger.warning("Malformed clusters `{}`".format(malformed))


class ClusterDescriptiveStats(object):
    """Container for population and cluster descriptive stats and
    statistical tests."""

    def __init__(
        self,
        data: pd.DataFrame,
        assignments: np.ndarray,
        weights: Optional[pd.Series] = None,
        names: Optional[List[str]] = None,
        categoricals: Optional[List[str]] = None,
        numericals: Optional[List[str]] = None,
    ):
        """Initialize analysis metadata, validate weights and initialize
        the descriptive stats object."""
        check_clustering(assignments, names)
        self._data = data
        self._labels = assignments
        self._names = names
        self._columns = sorted(data.columns.tolist())
        self._df = data.loc[:, self._columns]
        self._cat_cols = sorted(categoricals) if categoricals else self._columns
        self._num_cols = sorted(numericals) if numericals else self._columns
        self._masks = []

        labels = np.unique(assignments)
        self.n_clusters = len(labels)
        for label in labels:
            mask = np.equal(assignments, label)
            self._masks.append(mask)

        if weights is None:
            weights = pd.Series(
                np.ones_like(assignments, dtype=np.float32), index=data.index
            )
        self.sample_weight = weights
        self.cluster_weight_series = pd.Series(
            data=np.zeros(self.n_clusters), index=self._names, name="Cluster weight"
        )
        self.pop_stats = DescrStatsW(data.loc[:, self._num_cols], weights, ddof=1)
        # TODO: Percentage of missing mass on each attribute?
        self.cluster_stats = []
        for i, mask in enumerate(self._masks):
            # Compute cluster means on attributes of interest
            x = data.loc[mask, self._num_cols]
            s = DescrStatsW(x, weights=weights[mask], ddof=1)
            self.cluster_stats.append(s)
            self.cluster_weight_series[i] = s.nobs

    def run(self, output_dir=None, topk=None, transparent=False):
        """Compute stats, plot, write relevant files to disk"""
        num_stats_df = self.numerical_stats()
        categorical_stats_dict = self.categorical_stats(topk=topk)
        ttests_df = self.ttests()
        vtests_dict = self.vtests(topk=topk)
        if output_dir:
            logger.info("[Clustering stats] In {}".format(output_dir))
            self.visualize(output_dir, num_stats_df, transparent)
            self.write_to_disk(
                output_dir, ttests_df, num_stats_df, categorical_stats_dict, vtests_dict
            )
        return num_stats_df, categorical_stats_dict, ttests_df, vtests_dict

    def write_to_disk(
        self,
        output_dir,
        ttests_df=None,
        num_stats_df=None,
        categorical_stats_dict=None,
        vtests_dict=None,
    ):
        # Distribution of weight per cluster
        clst_weight = self.cluster_weight_series.name
        weight_pct_name = clst_weight + " (%)"
        weight_pct_df = pd.DataFrame(
            np.append(to_percent(self.cluster_weight_series.values), 100),
            index=self.cluster_weight_series.index.tolist() + ["Total"],
            columns=[weight_pct_name],
        )
        weight_df = pd.DataFrame(
            np.append(self.cluster_weight_series, self.cluster_weight_series.sum()),
            index=self.cluster_weight_series.index.tolist() + ["Total"],
            columns=[clst_weight],
        )
        weight_tbl = pd.concat([weight_df, weight_pct_df], axis=1)
        weight_tbl.sort_values(weight_pct_name, inplace=True, ascending=False)
        weight_tbl.to_latex(join(output_dir, "cluster_weight.tex"))

        if num_stats_df is not None:
            # Round cluster stats
            output_stats_df = num_stats_df.round(2).fillna(0.)
            output_stats_df.to_latex(join(output_dir, "cluster_stats.tex"))
            output_stats_df.to_csv(join(output_dir, "cluster_stats.csv"))
            output_stats_df.to_html(join(output_dir, "cluster_stats.html"))

        if categorical_stats_dict is not None:
            html = []
            for name, df in categorical_stats_dict.items():
                fp = join(output_dir, "{} categorical stats".format(name))
                df.to_latex(fp + ".tex", index=False)
                html.append("<h2>{}</h2>".format(name) + df.to_html())
            with open(join(output_dir, "ClusterStats.html"), mode="w") as f:
                f.write("<br/><br/>".join(html))

        if ttests_df is not None:
            # Round it
            ttests_df.to_latex(join(output_dir, "ttests.tex"))
            ttests_df.to_html(join(output_dir, "ttests.html"))

        if vtests_dict is not None:
            tests_html = []
            for name, tests in vtests_dict.items():
                tests_html.append("<h2>{}</h2>".format(name) + tests.to_html())
            with open(join(output_dir, "V-tests.html"), mode="w") as f:
                f.write("<br/><br/>".join(tests_html))
            """
            tests_path = join(output_dir, 'V-tests {}'.format(cluster_name))
            clst_tests.to_latex(tests_path + '.tex', index=False)
            clst_tests.to_csv(tests_path + '.csv')
            """

    def visualize(
        self,
        output_dir: str,
        stats_df: pd.DataFrame,
        transparent: bool,
        histo_cols: Tuple = (),
        weight_figsize=(16, 9),
        histo_figsize=(32, 32),
    ):
        """ Plot weights, numeric stats, """
        histo_cols = histo_cols or self._num_cols
        for mask, name in zip(self._masks, self._names):
            hist_path = join(output_dir, "Hist {}.png".format(name))
            data, nonull = remove_missing(
                self._df.loc[mask, histo_cols], return_mask=True, raise_error=False
            )
            weights = self.sample_weight.values[mask][nonull]
            data.hist(bins=100, figsize=histo_figsize, weights=weights)
            plt.savefig(hist_path, transparent=transparent)
            plt.close()
            plt.clf()

        fp = join(output_dir, "clusters_weight.png")
        self.cluster_weight_series.plot.bar(figsize=weight_figsize)
        plt.ylabel("No. households")
        plt.tight_layout()
        plt.savefig(fp, transparent=transparent)
        plt.close()
        plt.clf()

        fp = join(output_dir, "clusters_weight_pie.png")
        make_annotated_pie(self.cluster_weight_series, self.cluster_weight_series)
        plt.tight_layout()
        plt.savefig(fp, transparent=transparent)
        plt.close()
        plt.clf()

        if stats_df is not None:
            # Barplot for each attribute, showing global average
            n = len(self._num_cols)
            fig, ax = plt.subplots(
                n, 1, sharex=True, squeeze=False, figsize=(16, 9 * n)
            )
            ax = ax.ravel()
            for i, column in enumerate(self._num_cols):
                ylim = (20, 35) if column == "bmi" else None
                stats_df.loc[:, (column, "mean")].plot.bar(
                    yerr=stats_df.loc[:, (column, "std")],
                    title=column,
                    ylim=ylim,
                    ax=ax[i],
                )
                ax[i].axhline(stats_df.loc["Global", column]["mean"])
            fig.tight_layout()
            plt.savefig(join(output_dir, "Numerical attributes barplots.png"))
            plt.close()
            plt.clf()

            heatmap_path = join(output_dir, "stats_average_heatmap.png")
            fig, ax = plt.subplots(1, 1, figsize=(30, 30))
            normed_stats_df = normalize(
                stats_df.loc[:, (slice(None), "mean")], copy=False
            )
            heatmap(normed_stats_df, ax=ax, linewidths=.5, annot=True)
            plt.yticks(rotation=0)
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(heatmap_path, transparent=transparent)
            plt.title("Average quantities purchased in tons (cluster x category)")
            plt.close()
            plt.clf()

        # Boxplot for each attribute, showing global average
        n = len(self._num_cols)
        fig, ax = plt.subplots(n, 1, sharex=True, squeeze=False, figsize=(16, 9 * n))
        ax = ax.ravel()  # Only squeezing 2nd dim
        names_df = pd.DataFrame(
            np.array(self._names)[self._labels],
            index=self._data.index,
            columns=["Cluster"],
        )
        data_with_names = pd.concat([self._data, names_df], axis=1)
        for i, column in enumerate(self._num_cols):
            boxplot(x="Cluster", y=column, data=data_with_names, ax=ax[i])
        fig.tight_layout()
        plt.savefig(join(output_dir, "Numerical attributes boxplots.png"))
        plt.close()
        plt.clf()

    def numerical_stats(self, columns=None):
        """Combine cluster-wise and population statistics"""
        columns = columns or self._num_cols
        logger.info("[Numerical stats] Attributes\n{}".format(columns))
        stats_columns = pd.MultiIndex.from_product(
            [columns, ["mean", "std"]], names=["attributes", "stats"]
        )
        stats_df = pd.DataFrame(
            index=self._names + ["Global"], columns=stats_columns, dtype=np.float64
        )
        stats_df.loc["Global", (slice(None), "mean")] = self.pop_stats.mean
        stats_df.loc["Global", (slice(None), "std")] = self.pop_stats.std
        for i, (stats, name) in enumerate(zip(self.cluster_stats, self._names)):
            # sample_stats.data = np.ma.masked_array(
            # sample_df.values, mask=sample_df.isnull())
            stats_df.loc[name, (slice(None), "mean")] = stats.mean
            stats_df.loc[name, (slice(None), "std")] = stats.std
        return stats_df

    def categorical_stats(self, columns=None, topk=None):
        columns = columns or self._cat_cols
        df = self._df.loc[:, columns]
        clst_stats = dict()
        logger.info("Categorical stats on attributes\n{}".format(df.columns.tolist()))
        for name, mask in zip(self._names, self._masks):
            cluster_desc_df = self._categorical_stats(
                df.loc[mask, :], self.sample_weight[mask], topk
            )
            clst_stats[name] = cluster_desc_df
        clst_stats["Population"] = self._categorical_stats(df, self.sample_weight, topk)
        return clst_stats

    @staticmethod
    def _categorical_stats(df, weights, topk=None):
        df = df.apply(weighted_value_counts, axis=0, weights=weights, pretty=True)
        df = df.head(topk).fillna("-")
        return df

    def ttests(self, columns=None):
        """Compute t-tests for numerical variables
        Notes
        -----
        The df passed here has to have columns that are aligned with the
        `pop_stats` structure. For the time being can't let the user choose
        which columns to test on: if we only want a subset of columns we
        first have to resolve the corresponding indices in `pop_stats`."""
        columns = columns or self._num_cols
        return ttests(
            self._df.loc[:, columns], self.pop_stats, self.cluster_stats, self._names
        )

    def vtests(self, columns=None, topk=None):
        columns = columns or self._cat_cols
        """Compute value tests for categorical variables."""
        return vtests(
            self._df.loc[:, columns],
            self._labels,
            self.sample_weight,
            self._names,
            topk,
        )
