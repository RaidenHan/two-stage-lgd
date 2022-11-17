#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier, \
    HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, \
    mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.compose import TransformedTargetRegressor

from utils.transformer import *


def save_model_performance(model, prefix, X_test, y_test):
    """ Save the trained model and the model performance

    Parameters
    ----------
    model : sklearn.model_selection._search.GridSearchCV
        Sklearn machine learning model
    prefix : str
        Prefix of the files
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test response variable

    """

    est_name = model.best_estimator_.steps[1][0]
    label = prefix + '_' + est_name

    # Save the model
    dump(model, f'model/{label}.joblib')

    if est_name in ['logistic', 'svc', 'random_forest']:
        # Save the model coefficient or the feature importance
        if est_name in ['logistic', 'svc']:
            coef = pd.Series(dict(zip(
                X_test.columns, model.best_estimator_[est_name].coef_[0])),
                name=label)
            coef.to_csv(f'model/{label}_coef.csv')
        elif est_name == 'random_forest':
            feature_imp = pd.Series(dict(zip(
                X_test.columns,
                model.best_estimator_[est_name].feature_importances_)),
                name=label)
            feature_imp.to_csv(f'model/{label}_importance.csv')

        y_pred = model.predict(X_test)
        # Plot the confusion matrix
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        ax.set_title(' '.join(
            [s.capitalize() for s in label.split('_')]) + ' Result')
        fig.tight_layout()
        plt.savefig(f"model/{label}.png")

        # Save the classification report
        report = pd.DataFrame(
            classification_report(
                y_test, y_pred, output_dict=True, zero_division=0))
        report.to_csv(f"model/{label}_report.csv")

    elif est_name in ['linear', 'boosting']:
        if est_name == 'linear':
            # Save the model coefficient
            coef = pd.Series(dict(zip(
                X_test.columns,
                model.best_estimator_[est_name].regressor_.coef_)),
                name=label)
            coef.to_csv(f'model/{label}_coef.csv')

        # Save the statistics
        y_pred = model.predict(X_test)
        result = pd.Series({'MAE': mean_absolute_error(y_test, y_pred),
                            'RMSE': np.sqrt(mean_squared_error(
                                y_test, y_pred))}, name=label)
        result.to_csv(f"model/{label}_result.csv")

    return


def logistic_regression(X_train, y_train, label, c_grid):
    """ Fit a logistic regression model to predict a certain label in the
    response variable with l2 penalty and 5-fold cross-validation. The function
    will use AUC to choose the optimal hyperparameter.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Features
    y_train : pandas.Series
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

    y_train_class = y_train == label
    scaler = StandardScaler()
    logistic = LogisticRegression(max_iter=10000, random_state=1104)
    pipe = Pipeline([('scaler', scaler), ('logistic', logistic)])
    param_grid = {'logistic__C': c_grid}
    search = GridSearchCV(
        pipe, param_grid, scoring=['roc_auc', 'accuracy'],
        refit='roc_auc', return_train_score=True)
    search.fit(X_train, y_train_class)

    return search


def linear_svc(X_train, y_train, label, c_grid):
    """ Fit a linear support vector classifier to predict a certain label in
    the response variable with l2 penalty and 5-fold cross-validation. The
    function will use AUC to choose the optimal hyperparameter.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Features
    y_train : pandas.Series
        Original response variable
    label : float
        Positive class value
    c_grid : array-like
        Inverse of regularization strength

    Returns
    -------
    search : sklearn.model_selection._search.GridSearchCV
        Optimal linear support vector model

    """

    y_train_class = y_train == label
    scaler = StandardScaler()
    svc = LinearSVC(
        class_weight='balanced', random_state=1104, max_iter=100000)
    pipe = Pipeline([('scaler', scaler), ('svc', svc)])
    param_grid = {'svc__C': c_grid}
    search = GridSearchCV(pipe, param_grid, scoring=['roc_auc', 'accuracy'],
                          refit='roc_auc', return_train_score=True)
    search.fit(X_train, y_train_class)

    return search


def random_forest_clf(X_train, y_train, label, depth_grid, min_split_grid):
    """ Fit a random forest classifier to predict a certain label in the
    response variable with l2 penalty and 5-fold cross-validation. The function
    will use AUC to choose the optimal hyperparameter.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Features
    y_train : pandas.Series
        Original response variable
    label : float
        Positive class value
    depth_grid : array-like
        Maximum depth of the tree
    min_split_grid : array-like
        Minimum number of samples required to split an internal node

    Returns
    -------
    search : sklearn.model_selection._search.GridSearchCV
        Optimal random forest model

    """

    y_train_class = y_train == label
    scaler = StandardScaler()
    rf_clf = RandomForestClassifier(n_estimators=500, random_state=1104)
    pipe = Pipeline([('scaler', scaler), ('random_forest', rf_clf)])
    param_grid = {'random_forest__max_depth': depth_grid,
                  'random_forest__min_samples_split': min_split_grid}
    search = GridSearchCV(pipe, param_grid, scoring=['roc_auc', 'accuracy'],
                          refit='roc_auc', return_train_score=True)
    search.fit(X_train, y_train_class)

    return search


def choose_best_model(model_list, param_metric, metric):
    """ Given a set of models, compare the metric corresponding to the optimal
    hyperparameter measured by their param_metric, and select the optimal
    model.

    Parameters
    ----------
    model_list : list
        List of machine learning models with cross-validation results
    param_metric : str
        Metric for selecting hyperparameters
    metric : str
        Metric for selecting the optimal model

    Returns
    -------
    model : sklearn.model_selection._search.GridSearchCV
        Optimal model

    """

    scores = []
    for model in model_list:
        scores.append(
            model.cv_results_[f'mean_test_{metric}'][
                np.argmax(model.cv_results_[f'mean_test_{param_metric}'])])
    model = model_list[np.argmax(scores)]

    return model


def linear_regression(X_train, y_train):
    """ Fit a linear model with 5-fold cross-validation. The function will
    choose the model with the minimum MAE.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Features
    y_train : pandas.Series
        Original response variable

    Returns
    -------
    search : sklearn.model_selection._search.GridSearchCV
        Optimal linear regression model

    """

    scaler = StandardScaler()
    transformers = [logit_transformer(), beta_transformer(y_train),
                    probit_transformer(y_train)]
    lr = LinearRegression()
    regr = TransformedTargetRegressor(lr, check_inverse=False)
    pipe = Pipeline([('scaler', scaler), ('linear', regr)])
    param_grid = {'linear__transformer': transformers}
    search = GridSearchCV(pipe, param_grid, scoring='neg_mean_absolute_error',
                          return_train_score=True)
    search.fit(X_train, y_train)

    return search


def gradient_boosting(X_train, y_train, max_depth_grid, min_sample_grid):
    """ Fit a gradient boosting regression tree with 5-fold cross-validation.
    The function will choose the model with the minimum MAE.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Features
    y_train : pandas.Series
        Original response variable
    max_depth_grid : array-like
        Maximum depth of the tree
    min_sample_grid : array-like
        Minimum number of samples per leaf

    Returns
    -------
    search : sklearn.model_selection._search.GridSearchCV
        Optimal linear regression model

    """

    scaler = StandardScaler()
    transformers = [logit_transformer(), beta_transformer(y_train),
                    probit_transformer(y_train)]
    gbr = HistGradientBoostingRegressor(
        loss='absolute_error', learning_rate=0.02, max_iter=500,
        early_stopping=True, random_state=1104)
    regr = TransformedTargetRegressor(gbr, check_inverse=False)
    pipe = Pipeline([('scaler', scaler), ('boosting', regr)])
    param_grid = {'boosting__transformer': transformers,
                  'boosting__regressor__max_depth': max_depth_grid,
                  'boosting__regressor__min_samples_leaf': min_sample_grid}
    search = GridSearchCV(pipe, param_grid, scoring='neg_mean_absolute_error',
                          return_train_score=True)
    search.fit(X_train, y_train)

    return search
