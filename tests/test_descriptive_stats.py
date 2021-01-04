import pytest
import logging
from tempfile import mkdtemp

import numpy as np

from kantar.config import DEFAULT_DATA_DIR
from kantar.households import descriptive_num_cols, descriptive_cat_attrs
from kantar.households.cohort_manager import CohortManager
from kantar.households.cohort_manager import var_to_human
from kantar.cluster.metrics import ClusterDescriptiveStats
from kantar.postprocess import component_names

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG
)

k = 10
weights = None
cluster_names = component_names("Cluster", k)


@pytest.fixture
def cohort_mgr():
    return CohortManager(DEFAULT_DATA_DIR)


@pytest.fixture
def cohort_df(cohort_mgr):
    df = cohort_mgr.select(
        head=True, panelist=True, preprocess=True, active=True
    )
    cols = sorted(set(descriptive_cat_attrs + descriptive_num_cols))
    logging.debug(cols)
    df = df.loc[:, cols]
    df = cohort_mgr.columns_to_human(df)
    logging.debug("\n" + str(df.head()))
    return df


@pytest.fixture
def assignments(cohort_df):
    return np.random.randint(k, size=len(cohort_df))


def test_cm_descr(cohort_mgr, assignments):
    tmpdir = mkdtemp()
    logging.info(tmpdir)
    house_ids = cohort_mgr.get_cohort(active=True)
    cohort_mgr.describe_centroids(
        house_ids, assignments, cluster_names, tmpdir
    )


def test_setup(
    cohort_df,
    assignments,
    vtest_cols=descriptive_num_cols,
    ttest_cols=descriptive_cat_attrs,
):
    tmpdir = mkdtemp()
    logging.info(tmpdir)
    human_vtests = var_to_human(vtest_cols)
    human_ttests = var_to_human(ttest_cols)
    cd = ClusterDescriptiveStats(
        cohort_df,
        assignments,
        weights,
        cluster_names,
        num_cols=human_ttests,
        cat_cols=human_vtests,
    )
    cd.run(tmpdir)
