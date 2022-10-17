#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def plot_distribution(s, **kwargs):
    """ Plot the distribution of a numeric random variable

    @author: Zhong

    Parameters
    ----------
    s : Series
        Numeric variable's sample

    """

    plt.hist(s, 20, density=1)


def plot_right_skewed_trans(s, **kwargs):
    """ Plot the transformed distributions for a left-skewed variable by taking
    cube root, square root, and logarithm

    @author: Zhong

    Parameters
    ----------
    s : Series
        Left-skewed numeric variable's sample

    """

    plt.figure()
    plt.hist(s ** (1 / 3), 20, density=1)
    plt.show()

    plt.figure()
    plt.hist(s ** (1 / 2), 20, density=1)
    plt.show()

    plt.figure()
    plt.hist(np.log(s), 20, density=1)
    plt.show()
