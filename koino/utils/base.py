# coding=utf-8
import logging
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def jaccard(a: Union[set, np.ndarray], b: Union[set, np.ndarray]) -> float:
    size_union = len(b | a)
    size_inter = len(a & b)
    if not size_union:
        return 0
    else:
        return size_inter / size_union


def weighted_value_counts(
    serie: pd.Series, weights: Optional[pd.Series] = None, pretty: bool = False
) -> pd.Series:
    """ Compute the weighted counts for each unique value of (presumably
     categorical) Series.

     Parameters
     ----------
     serie: pandas Series
     weights: pandas Series
     pretty: bool

     Returns
     -------
     value_weight: pandas Series
     """
    if weights is None:
        return serie.value_counts()

    serie_name = serie.name or 0
    weights_name = weights.name or 1
    # logger.debug(serie_name)
    # logger.debug(weights_name)
    weights_groupby = pd.concat([serie, weights], axis=1).groupby(serie_name)
    # logger.debug('\n' + str(weights_groupby.sum()))
    value_weight = weights_groupby.sum().iloc[:, 0] / weights.sum()
    value_weight.sort_values(ascending=False, inplace=True)
    value_weight.dropna(inplace=True)
    if pretty:
        value_weight = pd.Series(
            [
                "{} ({:.3f})".format(val, percent)
                for val, percent in zip(value_weight.index, value_weight)
                if percent not in (None, np.nan, float("nan"))
            ],
            name=weights_name,
        )
    return value_weight


def remove_missing(
    df: pd.DataFrame, return_mask: bool = False, raise_error: bool = True
) -> pd.DataFrame:
    """ Sanity check in case we forgot something or did not impute
    missing values. Easiest solution is to remove corresponding rows,
    instead of filling with zeros which biases the analysis.

    Returns
    -------
    df: Complete dataframe
    mask: kept indices """
    missing = df.isnull()
    num_missing = missing.sum(axis=0)
    if num_missing.any():
        if raise_error:
            cols_missing = num_missing[num_missing != 0]
            logging.warning("Columns with missing values:\n{}".format(cols_missing))
        mask = ~missing.any(axis=1)
        df = df[mask]
    else:
        mask = np.ones(len(df), dtype=np.bool_)

    if return_mask:
        return df, mask
    return df


def to_percent(serie: Union[np.ndarray, pd.Series], precision: int = 3) -> np.ndarray:
    return np.round(100 * serie / serie.sum(), precision)


def weighted_groupby(
    series: pd.Series, weights: Union[np.ndarray, pd.Series]
) -> pd.Series:
    if isinstance(weights, np.ndarray):
        weights = pd.Series(weights, index=series.index)

    data = dict()
    for value, same_val_series in series.groupby(series):
        name = same_val_series.index[0]
        sample_weight = weights.loc[same_val_series.index].values.sum()
        data[name] = sample_weight
    series_ = pd.Series(data, name=series.name)
    series_ /= series_.sum()
    series_.sort_values(inplace=True, ascending=False)
    return series_


def describe(
    data: Union[pd.Series, pd.DataFrame],
    column: Optional[str] = None,
    sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        data = data.loc[:, column]
    else:
        assert isinstance(data, pd.Series)
    # Groupby doesn't handle NaNs
    data.fillna("Manquant", inplace=True)
    data = data.str.capitalize()
    description_df = data.value_counts()

    # Compute weighted stats for population
    if sample_weight is not None:
        sample_distrib = to_percent(weighted_groupby(data, sample_weight))
        # Make table with both counts and percentages
        values_distrib = pd.Series(sample_distrib, name=data.name + " (%)")
        description_df = pd.concat([description_df, values_distrib], axis=1)
        description_df.sort_values(values_distrib.name, inplace=True, ascending=False)

    return description_df
