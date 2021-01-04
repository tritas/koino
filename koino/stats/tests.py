import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats
from statsmodels.stats.weightstats import DescrStatsW

from ..utils.base import weighted_value_counts

logger = logging.getLogger(__name__)


def numeric_value_test(
    values: np.ndarray, cluster_mask: np.ndarray, epsilon: float = 1e-8
) -> float:
    """ Mu and sigma are computed on the whole dataset """
    n_samples = len(values)
    n = np.sum(cluster_mask)
    mu, sigma = np.mean(values), np.var(values)
    mu_cluster = np.mean(values[cluster_mask])
    stddev = np.sqrt(((n_samples - n) / (n_samples - 1)) * ((sigma ** 2) / n))
    test_value = (mu_cluster - mu) / (stddev + epsilon)
    return test_value


def categorical_value_test(
    tot_samples: int, cluster_samples: int, cat_samples: int, n_cat_cluster: int
) -> float:
    denom = (
        ((tot_samples - cluster_samples) / (tot_samples - 1))
        * (1 - cat_samples / tot_samples)
        * (cluster_samples * cat_samples / tot_samples)
    )
    test_value = n_cat_cluster - cluster_samples * cat_samples / tot_samples
    test_value /= np.sqrt(denom)
    return test_value


def ttests(
    df: pd.DataFrame,
    pop_stats: DescrStatsW,
    cluster_stats: List[DescrStatsW],
    cluster_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute test values from population and cluster statistics
    Parameters
    ----------
    df: pandas DataFrame
    pop_stats: statsmodels DescrStatsW
    cluster_stats: list of statsmodels DescrStatsW
    cluster_names: list

    Returns
    -------
    tests_df: pandas DataFrame of shape [n_clusters + 1, len(columns)]
    """
    columns = df.columns.tolist()
    logger.info("[t-tests] On data: {}".format(df.shape))
    logger.info("[t-tests] Attributes\n{}".format(columns))
    # MultiIndex can produce errors if levels not sorted?
    # pop_stats.data = np.ma.masked_array(df.values, mask=df.isnull())
    test_columns = pd.MultiIndex.from_product(
        [columns, ["p-value", "t-test"]], names=["attributes", "tests"]
    )
    tests_df = pd.DataFrame(index=cluster_names, columns=test_columns, dtype=np.float32)
    for i, (name, stats) in enumerate(zip(cluster_names, cluster_stats)):
        # Compute t-tests
        for column, pop_mean, pop_std, sample_mean, sample_std in zip(
            columns, pop_stats.mean, pop_stats.std, stats.mean, stats.std
        ):
            tval, pval = ttest_ind_from_stats(
                pop_mean,
                pop_std,
                pop_stats.nobs,
                sample_mean,
                sample_std,
                stats.nobs,
                equal_var=False,
            )
            tests_df.loc[name, column] = [pval.round(6), tval.round(2)]
    return tests_df


def vtests(
    df: pd.DataFrame,
    assignments: np.ndarray,
    weights: np.ndarray,
    cluster_names: List[str],
    topk: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute categorical v-tests for given cluster assignments.
    For each cluster, for each attribute, for each value, we
    compute the v-test between cluster value counts and global value counts.

    Parameters
    ----------
    df: pandas DataFrame
    assignments: array-like of shape [n_samples,]
    weights: array-like of shape [n_samples,], None
    cluster_names: list of size [n_clusters,], None
    output_dir: str, None
    topk: int, None

    Returns
    -------
    cluster_test_dfs: list of pandas DataFrames of size [n_clusters,]
    """
    logger.info("[V-tests] Data: {}".format(df.shape))
    logger.info("[V-tests] Attributes\n{}".format(df.columns.tolist()))
    total_weight = weights.sum()
    cluster_tests_dict = dict()

    for i, name in enumerate(cluster_names):
        mask = np.equal(assignments, i)
        cluster_weights = weights[mask]
        cluster_tot_weight = cluster_weights.sum()
        test_dfs = []
        for col in df.columns:
            # Compute weight of each attribute value
            val_weights = weighted_value_counts(df.loc[:, col], weights)
            # Here some attributes could be missing
            clst_val_weights = weighted_value_counts(
                df.loc[mask, col], cluster_weights
            ).dropna()
            col_tests = clst_val_weights.copy()
            col_tests.name = "v-test"
            for (attr_val, cluster_val_weight) in clst_val_weights.iteritems():
                test = categorical_value_test(
                    total_weight,
                    cluster_tot_weight,
                    val_weights[attr_val],
                    cluster_val_weight,
                )
                col_tests.loc[attr_val] = test
            col_tests = col_tests.sort_values(ascending=False)
            col_tests.index = pd.MultiIndex.from_product([[col], col_tests.index])
            if topk and len(col_tests) > topk:
                col_tests = col_tests.head(topk)
            test_dfs.append(pd.DataFrame(col_tests))
        cluster_tests_dict[name] = pd.concat(test_dfs)
    return cluster_tests_dict
