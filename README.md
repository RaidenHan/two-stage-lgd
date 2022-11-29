# Estimating Loss Given Default (LGD) with a Two-Stage Model

*Mentor: Raiden Han, Catherine Tang*

*Group Member: Xudong Chen, Abhilash Kalapatapu, Zitao Song, Zhong Tian*

## Abstract

Estimating loss given default (LGD) is a critical step in pricing credit products and managing credit risk. The project employed a two-stage approach to capture the boundary instances in Fannie Mae Single-Family Loan Performance Data. Along with loan and macroeconomic features, the method made use of three transformers and five cross-validated machine learning models. The outcome demonstrated that borrower-related variables were unimportant in estimating loss given default, and a two-stage model assisted in lowering the mean absolute error.

## Setup

In addition to the packages in requirements.txt, the project relies on PySpark to handle Fannie Mae's big data. Users who wish to use the data pre-processing script from this project should configure PySpark themselves and store the corresponding CSV files in the data/original_data directory.

## Structure

- Scripts
  - data_preprocessing.py: Preprocess the big data and extract useful information, North Carolina's defaulted loan data and macroeconomic variables in this project, into the data/primary_dataset.csv file.
  - feature_selection_eda.ipynb: Select features to put into the models and conduct exploratory data analysis.
  - modeling.py: Construct machine learning models and save the test results.
- directories
  - data: Extracted data for Pandas analysis and scikit-learn modeling.
  - docs: Presentation slides.
  - fig: Figures about feature distributions and relationships.
  - model: Used machine learning models and their performances on the test set.
  - utils: Other functions.