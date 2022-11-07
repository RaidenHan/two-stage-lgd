#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import date, timedelta
from functools import reduce
from operator import add
from os import listdir

import pandas as pd
import pandas_datareader.data as web
import pyspark.sql.functions as F
from pyspark.sql.types import *


def read_loan_csv(spark, filepath):
    """ Read the Fannie Mae single-family loan data with header

    @author: Raiden

    Parameters
    ----------
    spark : sparkSession
        User-specified Spark Session
    filepath : str
        Any valid string path is acceptable

    Returns
    -------
    df : pyspark.sql.DataFrame
        two-dimensional data structure with labeled axes

    """

    schema = StructType([
        StructField("POOL_ID", StringType(), True),
        StructField("LOAN_ID", StringType(), True),
        StructField("ACT_PERIOD", DateType(), True),
        StructField("CHANNEL", StringType(), True),
        StructField("SELLER", StringType(), True),
        StructField("SERVICER", StringType(), True),
        StructField("MASTER_SERVICER", StringType(), True),
        StructField("ORIG_RATE", FloatType(), True),
        StructField("CURR_RATE", FloatType(), True),
        StructField("ORIG_UPB", FloatType(), True),
        StructField("ISSUANCE_UPB", FloatType(), True),
        StructField("CURRENT_UPB", FloatType(), True),
        StructField("ORIG_TERM", ShortType(), True),
        StructField("ORIG_DATE", DateType(), True),
        StructField("FIRST_PAY", DateType(), True),
        StructField("LOAN_AGE", ShortType(), True),
        StructField("REM_MONTHS", ShortType(), True),
        StructField("ADJ_REM_MONTHS", ShortType(), True),
        StructField("MATR_DT", DateType(), True),
        StructField("OLTV", ShortType(), True),
        StructField("OCLTV", ShortType(), True),
        StructField("NUM_BO", ByteType(), True),
        StructField("DTI", ByteType(), True),
        StructField("CSCORE_B", ShortType(), True),
        StructField("CSCORE_C", ShortType(), True),
        StructField("FIRST_FLAG", StringType(), True),
        StructField("PURPOSE", StringType(), True),
        StructField("PROP", StringType(), True),
        StructField("NO_UNITS", ByteType(), True),
        StructField("OCC_STAT", StringType(), True),
        StructField("STATE", StringType(), True),
        StructField("MSA", StringType(), True),
        StructField("ZIP", StringType(), True),
        StructField("MI_PCT", FloatType(), True),
        StructField("PRODUCT", StringType(), True),
        StructField("PPMT_FLG", StringType(), True),
        StructField("IO", StringType(), True),
        StructField("FIRST_PAY_IO", DateType(), True),
        StructField("MNTHS_TO_AMTZ_IO", ShortType(), True),
        StructField("DLQ_STATUS", StringType(), True),
        StructField("PMT_HISTORY", StringType(), True),
        StructField("MOD_FLAG", StringType(), True),
        StructField("MI_CANCEL_FLAG", StringType(), True),
        StructField("Zero_Bal_Code", StringType(), True),
        StructField("ZB_DTE", DateType(), True),
        StructField("LAST_UPB", FloatType(), True),
        StructField("RPRCH_DTE", DateType(), True),
        StructField("CURR_SCHD_PRNCPL", FloatType(), True),
        StructField("TOT_SCHD_PRNCPL", FloatType(), True),
        StructField("UNSCHD_PRNCPL_CURR", FloatType(), True),
        StructField("LAST_PAID_INSTALLMENT_DATE", DateType(), True),
        StructField("FORECLOSURE_DATE", DateType(), True),
        StructField("DISPOSITION_DATE", DateType(), True),
        StructField("FORECLOSURE_COSTS", FloatType(), True),
        StructField("PROPERTY_PRESERVATION_AND_REPAIR_COSTS",
                    FloatType(), True),
        StructField("ASSET_RECOVERY_COSTS", FloatType(), True),
        StructField("MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS",
                    FloatType(), True),
        StructField("ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY",
                    FloatType(), True),
        StructField("NET_SALES_PROCEEDS", FloatType(), True),
        StructField("CREDIT_ENHANCEMENT_PROCEEDS", FloatType(), True),
        StructField("REPURCHASES_MAKE_WHOLE_PROCEEDS", FloatType(), True),
        StructField("OTHER_FORECLOSURE_PROCEEDS", FloatType(), True),
        StructField("NON_INTEREST_BEARING_UPB", FloatType(), True),
        StructField("PRINCIPAL_FORGIVENESS_AMOUNT", FloatType(), True),
        StructField("ORIGINAL_LIST_START_DATE", DateType(), True),
        StructField("ORIGINAL_LIST_PRICE", FloatType(), True),
        StructField("CURRENT_LIST_START_DATE", DateType(), True),
        StructField("CURRENT_LIST_PRICE", FloatType(), True),
        StructField("ISSUE_SCOREB", ShortType(), True),
        StructField("ISSUE_SCOREC", ShortType(), True),
        StructField("CURR_SCOREB", ShortType(), True),
        StructField("CURR_SCOREC", ShortType(), True),
        StructField("MI_TYPE", StringType(), True),
        StructField("SERV_IND", StringType(), True),
        StructField("CURRENT_PERIOD_MODIFICATION_LOSS_AMOUNT",
                    FloatType(), True),
        StructField("CUMULATIVE_MODIFICATION_LOSS_AMOUNT", FloatType(), True),
        StructField("CURRENT_PERIOD_CREDIT_EVENT_NET_GAIN_OR_LOSS",
                    FloatType(), True),
        StructField("CUMULATIVE_CREDIT_EVENT_NET_GAIN_OR_LOSS",
                    FloatType(), True),
        StructField("HOMEREADY_PROGRAM_INDICATOR", StringType(), True),
        StructField("FORECLOSURE_PRINCIPAL_WRITE_OFF_AMOUNT",
                    FloatType(), True),
        StructField("RELOCATION_MORTGAGE_INDICATOR", StringType(), True),
        StructField("ZERO_BALANCE_CODE_CHANGE_DATE", DateType(), True),
        StructField("LOAN_HOLDBACK_INDICATOR", StringType(), True),
        StructField("LOAN_HOLDBACK_EFFECTIVE_DATE", DateType(), True),
        StructField("DELINQUENT_ACCRUED_INTEREST", FloatType(), True),
        StructField("PROPERTY_INSPECTION_WAIVER_INDICATOR",
                    StringType(), True),
        StructField("HIGH_BALANCE_LOAN_INDICATOR", StringType(), True),
        StructField("ARM_5_YR_INDICATOR", StringType(), True),
        StructField("ARM_PRODUCT_TYPE", StringType(), True),
        StructField("MONTHS_UNTIL_FIRST_PAYMENT_RESET", ShortType(), True),
        StructField("MONTHS_BETWEEN_SUBSEQUENT_PAYMENT_RESET",
                    ShortType(), True),
        StructField("INTEREST_RATE_CHANGE_DATE", DateType(), True),
        StructField("PAYMENT_CHANGE_DATE", DateType(), True),
        StructField("ARM_INDEX", StringType(), True),
        StructField("ARM_CAP_STRUCTURE", StringType(), True),
        StructField("INITIAL_INTEREST_RATE_CAP", FloatType(), True),
        StructField("PERIODIC_INTEREST_RATE_CAP", FloatType(), True),
        StructField("LIFETIME_INTEREST_RATE_CAP", FloatType(), True),
        StructField("MARGIN", FloatType(), True),
        StructField("BALLOON_INDICATOR", StringType(), True),
        StructField("PLAN_NUMBER", StringType(), True),
        StructField("FORBEARANCE_INDICATOR", StringType(), True),
        StructField("HIGH_LOAN_TO_VALUE_HLTV_REFINANCE_OPTION_INDICATOR",
                    StringType(), True),
        StructField("DEAL_NAME", StringType(), True),
        StructField("RE_PROCS_FLAG", StringType(), True),
        StructField("ADR_TYPE", StringType(), True),
        StructField("ADR_COUNT", ShortType(), True),
        StructField("ADR_UPB", FloatType(), True)])
    df = spark.read.options(
        sep="|", dateFormat="MMyyyy").schema(schema).csv(filepath)

    return df


def find_default_data(df, disp_date=None, states=None):
    """ Find default data without repurchase and with a disposition date before
    a certain date

    @author: Abhilash, Raiden

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Single family loan data from Fannie Mae
    disp_date : str or None
        The loan's disposition date should be no later than the disp_date. The
        value is one year before today by default
    states : list
        States in the U.S. to be kept. All state data is retained by default

    Returns
    -------
    subset : DataFrame
        Filtered loan data

    """

    # Set disp_date's default value
    if disp_date is None:
        today = date.today()
        disp_date = today - timedelta(days=365)
        disp_date = disp_date.strftime('%Y-%m-%d')
    # Set states' default value
    if states is None:
        states = []
    # Set a list for default loans' zero balance codes
    default_code = ['02', '03', '09']
    # Subset the DataFrame
    subset = df.filter(df.Zero_Bal_Code.isin(default_code) &
                       (df.RE_PROCS_FLAG == 'N') &
                       (df.DISPOSITION_DATE < disp_date))
    if states:
        subset = subset.filter(subset.STATE.isin(states))

    return subset


def calculate_lgd(df):
    """ Calculate loss given default

    @author: Abhilash, Catherine, Raiden

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Single family loan data from Fannie Mae

    Returns
    -------
    df : pyspark.sql.DataFrame
        Single family loan data from Fannie Mae with loss given default

    """

    # Define lists of costs and proceeds
    costs_val = ["FORECLOSURE_COSTS", "PROPERTY_PRESERVATION_AND_REPAIR_COSTS",
                 "ASSET_RECOVERY_COSTS",
                 "MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS",
                 "ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY"]
    proceeds_val = ["NET_SALES_PROCEEDS", "CREDIT_ENHANCEMENT_PROCEEDS",
                    "REPURCHASES_MAKE_WHOLE_PROCEEDS",
                    "OTHER_FORECLOSURE_PROCEEDS"]
    # Fill null values in the costs and proceeds with 0
    df = df.na.fill(value=0, subset=costs_val + proceeds_val)
    # Calculate total costs and total proceeds
    df = df.withColumn(
        "DELINQUENT_ACCRUED_INTEREST",
        df.LAST_UPB * ((df.ORIG_RATE / 100 - 0.0035) / 12) * F.months_between(
            df.DISPOSITION_DATE, df.ACT_PERIOD))
    df = df.withColumn(
        "TOTAL_COST", reduce(add, [F.col(cost) for cost in costs_val]))
    df = df.withColumn(
        "TOTAL_PROCEED", reduce(
            add, [F.col(proceed) for proceed in proceeds_val]))
    df = df.withColumn("TOTAL_NET_LOSS",
                       df.LAST_UPB + df.DELINQUENT_ACCRUED_INTEREST +
                       df.TOTAL_COST - df.TOTAL_PROCEED)

    # Calculate LGD
    df = df.withColumn("LOSS_GIVEN_DEFAULT",
                       df.TOTAL_NET_LOSS / df.LAST_UPB)
    # Change LGD outside the boundary to endpoint values
    df = df.withColumn(
        "LOSS_GIVEN_DEFAULT", F.when(
            df.LOSS_GIVEN_DEFAULT > 1, 1).when(
            df.LOSS_GIVEN_DEFAULT < 0, 0).otherwise(df.LOSS_GIVEN_DEFAULT))

    return df


def merge_output_files(filepath, destination, **kwargs):
    """ Combine all csv files output by PySpark into one file

    Parameters
    ----------
    filepath : str
        The output path of PySpark
    destination : str
        Path object implementing a write() function

    Returns
    -------
    df : pandas.DataFrame
        Single family loan data from Fannie Mae with loss given default

    """

    filenames = listdir(filepath)
    csv_files = [filepath + '/' + filename
                 for filename in filenames if filename.endswith('.csv')]
    df_list = []
    for csv_file in csv_files:
        df_list.append(pd.read_csv(csv_file))
    df = pd.concat(df_list)
    df = df.reset_index(drop=True)
    df.to_csv(destination, **kwargs)

    return df


def get_fred_data(symbols,
                  start='1976-01-01', end=date.today().strftime('%Y-%m-%d')):
    """ Get monthly data for the given name from the St. Louis FED (FRED).
    Return all data from 1976 to the present by default. Null values will be
    filled using linear interpolation.

    Parameters
    ----------
    symbols : list
        FRED symbols
    start : str
        Start date of the data
    end : str
        End date of the data

    Returns
    -------
    df : pandas.DataFrame
        FRED data
    """

    symbol_list = []
    for symbol in symbols:
        s = web.DataReader(symbol, 'fred', start, end)
        s = s.resample('MS').first()
        symbol_list.append(s)
    df = pd.concat(symbol_list, axis=1)
    df = df.interpolate()

    return df


def merge_drop_features(loan_features, macro_features, drop_list):
    """ Merge the loan feature set and the macroeconomic feature set, calculate
    mark-to-market LTV and difference between interest rates, and drop
    selected features

    Parameters
    ----------
    loan_features : pandas.DataFrame
        Loan feature set
    macro_features : pandas.DataFrame
        Macroeconomic feature set
    drop_list : list
        Features to be dropped

    Returns
    -------
    df : pandas.DataFrame
       Primary feature set

    """

    # Merge the datasets
    df = pd.merge(loan_features, macro_features, how='left',
                  left_on='LAST_PAID_INSTALLMENT_DATE', right_index=True)
    df = pd.merge(df, macro_features[['CURR_HPI']].rename(
        columns={'CURR_HPI': 'ORIG_HPI'}), how='left',
                  left_on='ORIG_DATE', right_index=True)
    # Calculate mark-to-market LTV
    df['MTM_LTV'] = (df['LAST_UPB'] / ((df['ORIG_UPB'] / df['OLTV']) * (
            df['CURR_HPI'] / df['ORIG_HPI'])))
    # Calculate the difference between the original interest rate and the
    # current mortgage rate
    df['INT_DIFF'] = df['ORIG_RATE'] - df['MORTGAGE30US']
    df = df.drop(drop_list, axis=1)

    return df
