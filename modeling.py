#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import all packages needed
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.model import *

# The main function
def main():
    df = pd.read_csv("data/final_dataset.csv", index_col=0)
    X_train, X_test, y_train, y_test = train_test_split(
        pd.get_dummies(df.drop('LOSS_GIVEN_DEFAULT', axis=1), drop_first=True),
        df['LOSS_GIVEN_DEFAULT'], test_size=0.2, random_state=1104)

    # Predict for the LGD = 1 case
    logistic_1 = logistic_regression(
        X_train, y_train, label=1, c_grid=np.linspace(0.01, 0.03, 21))
    svc_1 = linear_svc(
        X_train, y_train, label=1, c_grid=np.linspace(0.0001, 0.0008, 21))
    random_forest_1 = random_forest_clf(
        X_train, y_train, label=1,
        depth_grid=range(5, 10), min_split_grid=range(20, 31, 2))
    # Save the models and choose the optimal model
    c1_models = [logistic_1, svc_1, random_forest_1]
    for c1_model in c1_models:
        save_model_performance(c1_model, 'class_1', X_test, y_test == 1)
    c1_model = choose_best_model(c1_models, 'roc_auc', 'accuracy')
    # Deliver the prediction to the second classification model
    c1_pred = c1_model.predict(X_test)
    y_pred = pd.Series(c1_pred, index=y_test.index, name=y_test.name).astype(
        float).replace(0, np.nan)
    X_test_resid, y_test_resid = X_test.loc[~c1_pred, :], y_test[~c1_pred]

    # Predict for the LGD = 0 case
    logistic_0 = logistic_regression(
        X_train, y_train, label=0, c_grid=np.linspace(0.24, 0.26, 21))
    svc_0 = linear_svc(
        X_train, y_train, label=0, c_grid=np.linspace(0.18, 0.22, 21))
    random_forest_0 = random_forest_clf(
        X_train, y_train, label=0,
        depth_grid=range(12, 18), min_split_grid=range(41, 56, 2))
    # Save the models and choose the optimal model
    c0_models = [logistic_0, svc_0, random_forest_0]
    for c0_model in c0_models:
        save_model_performance(
            c0_model, 'class_0', X_test_resid, y_test_resid == 0)
    c0_model = choose_best_model(c0_models, 'roc_auc', 'accuracy')
    # Deliver the prediction to the next stage
    c0_pred = c0_model.predict(X_test_resid)
    y_pred = y_pred.fillna(pd.Series(
        c0_pred, index=y_test_resid.index, name=y_test_resid.name).astype(
        float).replace(0, np.nan).replace(1, 0))
    X_test_resid, y_test_resid = (
        X_test_resid.loc[~c0_pred, :], y_test_resid[~c0_pred])

    # Regression
    train_reg_idx = y_train[(y_train > 0) & (y_train < 1)].index
    X_train_reg = X_train.loc[train_reg_idx, :]
    y_train_reg = y_train.loc[train_reg_idx]
    linear = linear_regression(X_train_reg, y_train_reg)
    gb_reg = gradient_boosting(
        X_train_reg, y_train_reg,
        max_depth_grid=range(8, 13), min_sample_grid=range(70, 81, 2))
    # Save the models and choose the optimal model
    reg_models = [linear, gb_reg]
    for reg_model in reg_models:
        save_model_performance(reg_model, 'reg', X_test_resid, y_test_resid)
    reg_model = choose_best_model(reg_models, 'score', 'score')
    # Finish the prediction
    y_pred = y_pred.fillna(pd.Series(
        reg_model.predict(X_test_resid),
        index=y_test_resid.index, name=y_test_resid.name))
    # Save the overall scores
    model_score = pd.Series({'MAE': mean_absolute_error(y_test, y_pred),
                             'RMSE': np.sqrt(mean_squared_error(
                                 y_test, y_pred))}, name='Score')
    model_score.to_csv("model/final_score.csv")

    return


if __name__ == '__main__':
    main()
