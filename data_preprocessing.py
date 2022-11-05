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
    # Create a list of primary features
    features = ['SELLER', 'CURR_RATE', 'LOAN_AGE', 'OLTV', 'CSCORE_B',
                'PURPOSE', 'OCC_STAT', 'MI_PCT', 'DLQ_STATUS', 'PMT_HISTORY',
                'PRINCIPAL_FORGIVENESS_AMOUNT', 'MI_TYPE']
    # Create a list of U.S. states of interest
    states = ['GA', 'FL']

    # Select default loans from certain states and calculate loss given default
    dataset = read_loan_csv(spark, filepath)
    default_data = find_default_data(dataset, states=states)
    default_data = calculate_lgd(default_data)
    # Keep and store primary features and the response variable
    if features:
        subset = default_data.select(features + ['LOSS_GIVEN_DEFAULT'])
    else:
        subset = default_data
    subset.write.csv(destination, header=True, mode='overwrite')

    # Close the PySpark session
    spark.stop()
    # Merge all output files
    merge_output_files(destination, "data/processed_dataset.csv")

    return


if __name__ == "__main__":
    main()
