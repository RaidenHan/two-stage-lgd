#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import beta, norm
from sklearn.preprocessing import FunctionTransformer


def logit_func(x):
    return np.log(x) - np.log(1 - x)


def logit_func_inv(x):
    return np.exp(x) / (np.exp(x) + 1)


def logit_transformer():
    """ Transform the variable using the fractional logit model

    Returns
    -------
    transformer : sklearn.preprocessing.FunctionTransformer

    """

    transformer = FunctionTransformer(
        logit_func, logit_func_inv, check_inverse=False)

    return transformer


def beta_func(x, a, b):
    return norm.ppf(beta.cdf(x, a, b))


def beta_func_inv(x, a, b):
    return beta.ppf(norm.cdf(x), a, b)


def beta_transformer(s):
    """ Transform the variable from a Beta distribution to a normal
    distribution

    Parameters
    ----------
    s : pandas.Series
        Variable's value in the training set

    Returns
    -------
    transformer : sklearn.preprocessing.FunctionTransformer

    """

    a, b, *_ = beta.fit(s, floc=0, fscale=1)
    params = {'a': a, 'b': b}
    transformer = FunctionTransformer(
        beta_func, beta_func_inv, check_inverse=False,
        kw_args=params, inv_kw_args=params)

    return transformer


def probit_func(x, s):
    n = s.shape[0] + 2
    return norm.ppf((np.argsort(np.argsort(x)) + 1) / n)


def probit_func_inv(x, s):
    return np.quantile(s, norm.cdf(x))


def probit_transformer(s):
    """ Transform the variable using the Probit transformation

    Parameters
    ----------
    s : pandas.Series
        Variable's value in the training set

    Returns
    -------
    transformer : sklearn.preprocessing.FunctionTransformer

    """

    params = {'s': s}
    transformer = FunctionTransformer(
        probit_func, probit_func_inv, check_inverse=False,
        kw_args=params, inv_kw_args=params)

    return transformer
