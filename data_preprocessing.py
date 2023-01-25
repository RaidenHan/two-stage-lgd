#!/usr/bin/env python
# -*- coding: utf-8 -*-
# importing all packages, including pyspark
import os
import sys

import pyspark.pandas as ps
from pyspark.sql import SparkSession

from utils.data_preprocessing import *


def main():
    # Prepare the environment for Windows operating systems
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    # Create a new Spark session
    spark = SparkSession.builder.master(
        "local[4]").appName("my_app").getOrCreate()

    # Configurate the Spark session
    ps.set_option("compute.default_index_type", "distributed")
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", True)

    # Set the file paths to the original datasets and the destination
    filepath = "data/original_data"
    destination = "data/processed_data"

    # Create lists of primary features
    macro_features = ['DFF', 'NCUR', 'NCSTHPI', 'MORTGAGE30US']
    loan_features = ['ORIG_RATE', 'ORIG_UPB', 'ORIG_DATE', 'OLTV', 'DTI',
                     'CSCORE_B', 'PURPOSE', 'PROP', 'NO_UNITS', 'OCC_STAT',
                     'LAST_UPB', 'LAST_PAID_INSTALLMENT_DATE']
    # Create a list of U.S. states of interest
    states = ['NC']

    # Select default loans from certain states and calculate loss given default
    loan_data = read_loan_csv(spark, filepath)
    default_data = find_default_data(loan_data, states=states)
    default_data = calculate_lgd(default_data)
    # Keep and store primary features and the response variable
    try:
        subset = default_data.select(loan_features + ['LOSS_GIVEN_DEFAULT'])
    except NameError:
        subset = default_data
    subset.write.csv(destination, header=True, mode='overwrite')
    # Close the PySpark session
    spark.stop()

    # Merge all output files
    loan_output = merge_output_files(destination, "data/extracted_loan.csv")
    loan_output[['ORIG_DATE', 'LAST_PAID_INSTALLMENT_DATE']] = loan_output[[
        'ORIG_DATE', 'LAST_PAID_INSTALLMENT_DATE']].apply(pd.to_datetime)
    # Get macroeconomic variables
    macro_df = get_fred_data(macro_features)
    macro_df = macro_df.rename(
        columns={'NCUR': 'UNRATE', 'NCSTHPI': 'CURR_HPI'})
    # Combine all features and save the primary feature set
    df = merge_drop_features(loan_output, macro_df, drop_list=[
        'ORIG_RATE', 'ORIG_DATE', 'LAST_PAID_INSTALLMENT_DATE', 'ORIG_HPI'])
    # Save the dataset
    df.to_csv("data/primary_dataset.csv")

    return


if __name__ == "__main__":
    main()
