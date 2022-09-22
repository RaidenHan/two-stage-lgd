import pandas as pd


def find_default_data(df, disp_date=None, states=None):
    """ Find default data without repurchase and with a disposition date before
    a certain date

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

    pass


def calculate_lgd(df):
    """ Calculate loss given default

    Parameters
    ----------
    df : DataFrame
        Single family loan data from Fannie Mae

    Returns
    -------
    df_new : DataFrame
        Single family loan data from Fannie Mae with loss given default

    """

    pass
