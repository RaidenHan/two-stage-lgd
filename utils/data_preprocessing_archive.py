#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import date, timedelta

import numpy as np
import pandas as pd


def read_loan_csv(filepath, **kwargs):
    """ Read the Fannie Mae single-family loan data with header

    @author: Raiden

    Parameters
    ----------
    filepath : str
        Any valid string path is acceptable

    Returns
    -------
    df : DataFrame
        two-dimensional data structure with labeled axes

    """

    column_names = ["POOL_ID", "LOAN_ID", "ACT_PERIOD", "CHANNEL", "SELLER",
                    "SERVICER", "MASTER_SERVICER", "ORIG_RATE", "CURR_RATE",
                    "ORIG_UPB", "ISSUANCE_UPB", "CURRENT_UPB", "ORIG_TERM",
                    "ORIG_DATE", "FIRST_PAY", "LOAN_AGE", "REM_MONTHS",
                    "ADJ_REM_MONTHS", "MATR_DT", "OLTV", "OCLTV", "NUM_BO",
                    "DTI", "CSCORE_B", "CSCORE_C", "FIRST_FLAG", "PURPOSE",
                    "PROP", "NO_UNITS", "OCC_STAT", "STATE", "MSA", "ZIP",
                    "MI_PCT", "PRODUCT", "PPMT_FLG", "IO", "FIRST_PAY_IO",
                    "MNTHS_TO_AMTZ_IO", "DLQ_STATUS", "PMT_HISTORY",
                    "MOD_FLAG", "MI_CANCEL_FLAG", "Zero_Bal_Code", "ZB_DTE",
                    "LAST_UPB", "RPRCH_DTE", "CURR_SCHD_PRNCPL",
                    "TOT_SCHD_PRNCPL", "UNSCHD_PRNCPL_CURR",
                    "LAST_PAID_INSTALLMENT_DATE", "FORECLOSURE_DATE",
                    "DISPOSITION_DATE", "FORECLOSURE_COSTS",
                    "PROPERTY_PRESERVATION_AND_REPAIR_COSTS",
                    "ASSET_RECOVERY_COSTS",
                    "MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS",
                    "ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY",
                    "NET_SALES_PROCEEDS", "CREDIT_ENHANCEMENT_PROCEEDS",
                    "REPURCHASES_MAKE_WHOLE_PROCEEDS",
                    "OTHER_FORECLOSURE_PROCEEDS", "NON_INTEREST_BEARING_UPB",
                    "PRINCIPAL_FORGIVENESS_AMOUNT", "ORIGINAL_LIST_START_DATE",
                    "ORIGINAL_LIST_PRICE", "CURRENT_LIST_START_DATE",
                    "CURRENT_LIST_PRICE", "ISSUE_SCOREB", "ISSUE_SCOREC",
                    "CURR_SCOREB", "CURR_SCOREC", "MI_TYPE", "SERV_IND",
                    "CURRENT_PERIOD_MODIFICATION_LOSS_AMOUNT",
                    "CUMULATIVE_MODIFICATION_LOSS_AMOUNT",
                    "CURRENT_PERIOD_CREDIT_EVENT_NET_GAIN_OR_LOSS",
                    "CUMULATIVE_CREDIT_EVENT_NET_GAIN_OR_LOSS",
                    "HOMEREADY_PROGRAM_INDICATOR",
                    "FORECLOSURE_PRINCIPAL_WRITE_OFF_AMOUNT",
                    "RELOCATION_MORTGAGE_INDICATOR",
                    "ZERO_BALANCE_CODE_CHANGE_DATE", "LOAN_HOLDBACK_INDICATOR",
                    "LOAN_HOLDBACK_EFFECTIVE_DATE",
                    "DELINQUENT_ACCRUED_INTEREST",
                    "PROPERTY_INSPECTION_WAIVER_INDICATOR",
                    "HIGH_BALANCE_LOAN_INDICATOR", "ARM_5_YR_INDICATOR",
                    "ARM_PRODUCT_TYPE", "MONTHS_UNTIL_FIRST_PAYMENT_RESET",
                    "MONTHS_BETWEEN_SUBSEQUENT_PAYMENT_RESET",
                    "INTEREST_RATE_CHANGE_DATE", "PAYMENT_CHANGE_DATE",
                    "ARM_INDEX", "ARM_CAP_STRUCTURE",
                    "INITIAL_INTEREST_RATE_CAP", "PERIODIC_INTEREST_RATE_CAP",
                    "LIFETIME_INTEREST_RATE_CAP", "MARGIN",
                    "BALLOON_INDICATOR", "PLAN_NUMBER",
                    "FORBEARANCE_INDICATOR",
                    "HIGH_LOAN_TO_VALUE_HLTV_REFINANCE_OPTION_INDICATOR",
                    "DEAL_NAME", "RE_PROCS_FLAG", "ADR_TYPE", "ADR_COUNT",
                    "ADR_UPB"]
    column_types = ["string", "string", "string", "category", "string",
                    "string", "string", "float32", "float32", "float32",
                    "float32", "float32", "Int64", "string", "string",
                    "Int64", "Int64", "Int64", "string", "Int64", "Int64",
                    "Int64", "Int64", "Int64", "Int64", "category", "category",
                    "category", "Int64", "category", "category", "string",
                    "string", "float32", "category", "category", "category",
                    "string", "Int64", "string", "string", "category",
                    "category", "category", "string", "float32", "string",
                    "float32", "float32", "float32", "string", "string",
                    "string", "float32", "float32", "float32", "float32",
                    "float32", "float32", "float32", "float32", "float32",
                    "float32", "float32", "string", "float32", "string",
                    "float32", "Int64", "Int64", "Int64", "Int64", "category",
                    "category", "float32", "float32", "float32", "float32",
                    "category", "float32", "category", "string", "category",
                    "string", "float32", "category", "category", "category",
                    "string", "Int64", "Int64", "string", "string", "string",
                    "string", "float32", "float32", "float32", "float32",
                    "category", "string", "category", "category", "string",
                    "category", "category", "Int64", "float"]
    column_types = dict(zip(column_names, column_types))
    date_col = ["ACT_PERIOD", "ORIG_DATE", "FIRST_PAY", "MATR_DT",
                "FIRST_PAY_IO", "ZB_DTE", "RPRCH_DTE",
                "LAST_PAID_INSTALLMENT_DATE", "FORECLOSURE_DATE",
                "DISPOSITION_DATE", "ORIGINAL_LIST_START_DATE",
                "CURRENT_LIST_START_DATE", "ZERO_BALANCE_CODE_CHANGE_DATE",
                "LOAN_HOLDBACK_EFFECTIVE_DATE", "INTEREST_RATE_CHANGE_DATE",
                "PAYMENT_CHANGE_DATE"]
    date_parser = lambda x: pd.to_datetime(x, format="%m%Y", errors='coerce')
    df = pd.read_csv(filepath, sep="|", names=column_names, dtype=column_types,
                     parse_dates=date_col, date_parser=date_parser, **kwargs)

    return df


def find_default_data(df, disp_date=None, states=None):
    """ Find default data without repurchase and with a disposition date before
    a certain date

    @author: Abhilash, Raiden

    Parameters
    ----------
    df : DataFrame
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
    subset = df.loc[(df['Zero_Bal_Code'].isin(default_code)) &
                    (df['RE_PROCS_FLAG'] == 'N') &
                    (df['DISPOSITION_DATE'] < disp_date), :]
    if states:
        subset = subset.loc[subset['STATE'].isin(states), :]

    return subset


def calculate_lgd(df):
    """ Calculate loss given default

    @author: Abhilash, Catherine

    Parameters
    ----------
    df : DataFrame
        Single family loan data from Fannie Mae

    Returns
    -------
    df_new : DataFrame
        Single family loan data from Fannie Mae with loss given default

    """

    df_new = df.copy()

    delinquent_accrued_interest = df['LAST_UPB'] * (
            (df['ORIG_RATE'] / 100 - 0.0035) / 12) * np.round(
        (df['DISPOSITION_DATE'] - df['ACT_PERIOD']) / np.timedelta64(1, 'M'))

    total_costs = (df["FORECLOSURE_COSTS"] +
                   df["PROPERTY_PRESERVATION_AND_REPAIR_COSTS"] +
                   df["ASSET_RECOVERY_COSTS"] +
                   df["MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS"] +
                   df["ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY"])

    total_proceeds = (df["NET_SALES_PROCEEDS"] +
                      df["CREDIT_ENHANCEMENT_PROCEEDS"] +
                      df["REPURCHASES_MAKE_WHOLE_PROCEEDS"] +
                      df["OTHER_FORECLOSURE_PROCEEDS"])

    # Calculate LGD
    total_net_loss = (df_new["LAST_UPB"] + delinquent_accrued_interest +
                      total_costs - total_proceeds)
    df_new['LGD'] = total_net_loss / df_new['LAST_UPB']

    return df_new
