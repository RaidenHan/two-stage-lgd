#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def logistic_regression(X_train, y_train, label, c_grid):
    """ Fit a logistic regression model to predict a certain label in the
    response variable with l2 penalty and 5-fold cross-validation. The function
    will use AUC to choose the optimal hyperparameters.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Features
    y_train : pandas.DataFrame
        Original response variable
    label : float
        Positive class value
    c_grid : array-like
        Inverse of regularization strength

    Returns
    -------
    search : sklearn.model_selection._search.GridSearchCV
        Optimal logistic regression model

    """

    y_train = y_train == label
    scaler = StandardScaler()
    logistic = LogisticRegression(max_iter=10000, random_state=1104)
    pipe = Pipeline([('scaler', scaler), ('logistic', logistic)])
    param_grid = {'logistic__C': c_grid}
    search = GridSearchCV(pipe, param_grid, scoring='roc_auc')
    search.fit(X_train, y_train)

    return search
