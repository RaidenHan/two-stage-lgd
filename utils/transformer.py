#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from scipy.stats import beta, norm


def beta_transformer(s):
    """ Transform the variable from a Beta distribution to a normal
    distribution

    Parameters
    ----------
    s : pandas.Series
        Variable's value

    Returns
    -------
    s_norm : pandas.Series
        Transformed variable's value
    a : float
        Fitted Beta distribution's loc parameter
    b : float
        Fitted Beta distribution's scale parameter

    """

    a, b, *_ = beta.fit(s, floc=0, fscale=1)
    s_norm = norm.ppf(beta.cdf(s, a, b))
    s_norm = pd.Series(s_norm, name=s.name)

    return s_norm, a, b
