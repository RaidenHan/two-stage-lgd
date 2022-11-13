#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split

from utils.model import *


def main():
    df = pd.read_csv("data/final_dataset.csv", index_col=0)
    X_train, X_test, y_train, y_test = train_test_split(
        pd.get_dummies(df.drop('LOSS_GIVEN_DEFAULT', axis=1), drop_first=True),
        df['LOSS_GIVEN_DEFAULT'], test_size=0.2, random_state=1104)
    # Predict for the LGD = 1 case
    logistic_1 = logistic_regression(
        X_train, y_train, label=1, c_grid=np.linspace(0.01, 0.03, 21))
    save_model_performance(logistic_1, 'class_1', X_test, y_test == 1)
    svc_1 = linear_svc(
        X_train, y_train, label=1, c_grid=np.linspace(0.0001, 0.0008, 21))
    save_model_performance(svc_1, 'class_1', X_test, y_test == 1)
    random_forest_1 = random_forest_clf(
        X_train, y_train, label=1,
        depth_grid=range(5, 10), min_split_grid=range(20, 31, 2))
    save_model_performance(random_forest_1, 'class_1', X_test, y_test == 1)

    return


if __name__ == '__main__':
    main()