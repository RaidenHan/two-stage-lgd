#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
        StructField("FIRST_FLAG", BooleanType(), True),
        StructField("PURPOSE", StringType(), True),
        StructField("PROP", StringType(), True),
        StructField("NO_UNITS", ByteType(), True),
        StructField("OCC_STAT", StringType(), True),
        StructField("STATE", StringType(), True),
        StructField("MSA", StringType(), True),
        StructField("ZIP", StringType(), True),
        StructField("MI_PCT", FloatType(), True),
        StructField("PRODUCT", StringType(), True),
        StructField("PPMT_FLG", BooleanType(), True),
        StructField("IO", BooleanType(), True),
        StructField("FIRST_PAY_IO", DateType(), True),
        StructField("MNTHS_TO_AMTZ_IO", ShortType(), True),
        StructField("DLQ_STATUS", StringType(), True),
        StructField("PMT_HISTORY", StringType(), True),
        StructField("MOD_FLAG", BooleanType(), True),
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
        StructField("SERV_IND", BooleanType(), True),
        StructField("CURRENT_PERIOD_MODIFICATION_LOSS_AMOUNT",
                    FloatType(), True),
        StructField("CUMULATIVE_MODIFICATION_LOSS_AMOUNT", FloatType(), True),
        StructField("CURRENT_PERIOD_CREDIT_EVENT_NET_GAIN_OR_LOSS",
                    FloatType(), True),
        StructField("CUMULATIVE_CREDIT_EVENT_NET_GAIN_OR_LOSS",
                    FloatType(), True),
        StructField("HOMEREADY_PROGRAM_INDICATOR", BooleanType(), True),
        StructField("FORECLOSURE_PRINCIPAL_WRITE_OFF_AMOUNT",
                    FloatType(), True),
        StructField("RELOCATION_MORTGAGE_INDICATOR", BooleanType(), True),
        StructField("ZERO_BALANCE_CODE_CHANGE_DATE", DateType(), True),
        StructField("LOAN_HOLDBACK_INDICATOR", BooleanType(), True),
        StructField("LOAN_HOLDBACK_EFFECTIVE_DATE", DateType(), True),
        StructField("DELINQUENT_ACCRUED_INTEREST", FloatType(), True),
        StructField("PROPERTY_INSPECTION_WAIVER_INDICATOR",
                    StringType(), True),
        StructField("HIGH_BALANCE_LOAN_INDICATOR", BooleanType(), True),
        StructField("ARM_5_YR_INDICATOR", BooleanType(), True),
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
        StructField("BALLOON_INDICATOR", BooleanType(), True),
        StructField("PLAN_NUMBER", StringType(), True),
        StructField("FORBEARANCE_INDICATOR", StringType(), True),
        StructField("HIGH_LOAN_TO_VALUE_HLTV_REFINANCE_OPTION_INDICATOR",
                    BooleanType(), True),
        StructField("DEAL_NAME", StringType(), True),
        StructField("RE_PROCS_FLAG", BooleanType(), True),
        StructField("ADR_TYPE", StringType(), True),
        StructField("ADR_COUNT", ShortType(), True),
        StructField("ADR_UPB", FloatType(), True)])
    df = spark.read.options(
        sep="|", dateFormat="MMyyyy").schema(schema).csv(filepath)

    return df
