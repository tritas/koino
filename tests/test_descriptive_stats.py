import pandas as pd
import pytest
import logging

import numpy as np

from koino.stats.descriptive import ClusterDescriptiveStats

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG
)

k = 10
weights = None
cluster_names = [f"Cluster.{i:02}" for i in range(k)]


@pytest.fixture
def numeric_cols():
    return ["x", "y", "z"]


@pytest.fixture
def categ_cols():
    return ["a", "b", "c"]


@pytest.fixture
def cohort_df(categ_cols, numeric_cols):
    cols = categ_cols, numeric_cols
    # TODO: Generate face data
    return pd.DataFrame(columns=cols)


@pytest.fixture
def assignments(cohort_df):
    return np.random.randint(k, size=len(cohort_df))


def test_setup(
    cohort_df,
    assignments,
    descriptive_num_cols,
    descriptive_cat_attrs,
):
    logging.info(tmpdir)
    human_vtests = list(map(str.upper, descriptive_cat_attrs))
    human_ttests = list(map(str.upper, descriptive_num_cols))
    cd = ClusterDescriptiveStats(
        cohort_df,
        assignments,
        weights,
        cluster_names,
        num_cols=human_ttests,
        cat_cols=human_vtests,
    )
    cd.run(tmpdir)
