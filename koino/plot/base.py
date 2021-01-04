# coding=utf-8
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..utils.base import to_percent
from ..utils.base import weighted_groupby


def make_annotated_pie(
    serie: pd.Series,
    sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
    figsize: Tuple[Union[int, float]] = (16, 16),
) -> None:
    """Compute a pie chart annotated with category counts and weights.

    Parameters
    ----------
    serie: pandas Series
    sample_weight: pandas Series (optional)
    figsize: tuple
    """
    if sample_weight is not None:
        serie_distrib = weighted_groupby(serie, sample_weight)
    else:
        serie_distrib = serie.value_counts()

    serie_distrib = to_percent(serie_distrib, 1)
    labels = []
    for label, val in zip(serie_distrib.index, serie_distrib.values):
        if isinstance(label, str):
            label = label.capitalize()
        label = "{} ({}%)".format(label, val)
        labels.append(label)
    serie_distrib.index = labels

    plt.figure(figsize=figsize)
    plt.title(serie.name)
    serie_distrib.plot(kind="pie")
