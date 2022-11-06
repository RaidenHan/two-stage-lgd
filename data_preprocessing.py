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
    merge_output_files(destination, "data/processed_dataset.csv")

    return


if __name__ == "__main__":
    main()
