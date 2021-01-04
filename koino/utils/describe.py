# coding=utf-8
import logging
from functools import partial
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def cns_str(attr: str) -> str:
    return "{} consensus".format(attr)


def name_by_tfidf(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    fun = partial(series_tfidf_name, **kwargs)
    return pd.DataFrame(df.apply(fun)).T


def approx_names(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """ Takes top-k cluster members to infer cluster attributes """
    df = pd.concat(map(name_by_tfidf, dfs), ignore_index=True, copy=False)
    labels = ["groupe", "sousgroupe"]
    df.loc[:, labels] = df.loc[:, labels].fillna("Inconnu")
    return df


def cluster_names(
    reference: pd.DataFrame,
    assignments: np.ndarray,
    label_columns: Union[Tuple[str], List[str]] = ("groupe", "sousgroupe"),
) -> pd.DataFrame:
    """Uses all cluster members to infer centroid attributes,
    weighted by TF-IDF."""

    cluster_labels = np.unique(assignments)
    dfs = []
    for label in cluster_labels:
        cluster_ref = reference[assignments == label]
        cluster_df = name_by_tfidf(cluster_ref)
        dfs.append(cluster_df)
    df = pd.concat(dfs, ignore_index=True, copy=False)
    # TODO: Resolve method `df = clean_df(df)`
    # TODO: Move the label columns out of this function
    df.loc[:, label_columns] = df.loc[:, label_columns].fillna("Inconnu")
    df.index = cluster_labels
    return df


def series_tfidf_name(
    series: pd.Series, thr: float = 0.5, fill_value: str = "Inconnu"
) -> str:
    series = series.fillna(fill_value)
    words = series.unique()
    vocabulary = dict(zip(words, range(series.nunique())))
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        lowercase=False,
        vocabulary=vocabulary,
        smooth_idf=True,
        norm="l1",
        sublinear_tf=True,
    )
    try:
        weights = vectorizer.fit_transform(series)
    except BaseException:
        logger.warning(vocabulary)
        raise
    feat_mean_weight = weights.mean(0).A.ravel()
    candidate = feat_mean_weight.argmax()
    feature_name = None
    if words[candidate] != fill_value and feat_mean_weight[candidate] > thr:
        feature_name = words[candidate]
    return feature_name


def infer_cluster_names(
    consensus_df: pd.DataFrame,
    majority_thr: float,
    column: Optional[str] = None,
    sep: str = ":",
) -> List[str]:
    """ Only check n_seed neighbors: if they all agree take
    vote, otherwise take closest one's label.

    Parameters
    ----------
    consensus_df: pandas DataFrame
    majority_thr: float
    column: str, None
    sep: str

    Returns
    -------
    cluster_names: list
    """

    def not_score(col):
        return not col.endswith("consensus")

    attrib_cols = list(filter(not_score, consensus_df.columns))
    attributes = [column] if column else attrib_cols
    cluster_names = consensus_df.index.values
    for i, (name, row) in enumerate(consensus_df.iterrows()):
        bits = []
        for attr in attributes:
            if row.loc[cns_str(attr)] > majority_thr:
                value = row.loc[attr]
                if attr in ("groupe", "sousgroupe"):
                    bits.append(value)
                elif value == "Fabricant Non Trouve":
                    bits.append("Sans fabricant")
                elif value == "Marque Non Trouvee":
                    bits.append("Sans marque")
                elif attr in ("bio", "mdd") and value == "Oui":
                    bits.append(attr)
                else:
                    value = value.replace("Groupe ", "").replace("GPE ", "")
                    bits.append(sep.join([attr, value.capitalize()]))
        infered_name = ", ".join(bits)
        if infered_name:
            cluster_names[i] = infered_name
    logger.info("Centroid infered names:\n{}".format(cluster_names[:-1]))
    return cluster_names[:-1]
