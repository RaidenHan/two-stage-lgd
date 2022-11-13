#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import reduce

import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


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
    y_test : pandas.DataFrame
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

    return


def logistic_regression(X_train, y_train, label, c_grid):
    """ Fit a logistic regression model to predict a certain label in the
    response variable with l2 penalty and 5-fold cross-validation. The function
    will use AUC to choose the optimal hyper-parameters.

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
    search = GridSearchCV(
        pipe, param_grid, scoring=['roc_auc', 'accuracy'],
        refit='roc_auc', return_train_score=True)
    search.fit(X_train, y_train)

    return search


def linear_svc(X_train, y_train, label, c_grid):
    """ Fit a linear support vector classifier to predict a certain label in the
    response variable with l2 penalty and 5-fold cross-validation. The function
    will use AUC to choose the optimal hyper-parameters.

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
        Optimal linear support vector model

    """

    y_train = y_train == label
    scaler = StandardScaler()
    svc = LinearSVC(class_weight='balanced', random_state=1104)
    pipe = Pipeline([('scaler', scaler), ('svc', svc)])
    param_grid = {'svc__C': c_grid}
    search = GridSearchCV(pipe, param_grid, scoring=['roc_auc', 'accuracy'],
        refit='roc_auc', return_train_score=True)
    search.fit(X_train, y_train)

    return search


def random_forest_clf(X_train, y_train, label, depth_grid, min_split_grid):
    """ Fit a random forest classifier to predict a certain label in the
    response variable with l2 penalty and 5-fold cross-validation. The function
    will use AUC to choose the optimal hyper-parameters.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Features
    y_train : pandas.DataFrame
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

    y_train = y_train == label
    scaler = StandardScaler()
    rf_clf = RandomForestClassifier(n_estimators=500, random_state=1104)
    pipe = Pipeline([('scaler', scaler), ('random_forest', rf_clf)])
    param_grid = {'random_forest__max_depth': depth_grid,
                  'random_forest__min_samples_split': min_split_grid}
    search = GridSearchCV(pipe, param_grid, scoring=['roc_auc', 'accuracy'],
        refit='roc_auc', return_train_score=True)
    search.fit(X_train, y_train)

    return search
