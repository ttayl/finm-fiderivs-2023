"""Loads historical Treasury yields data from

Refet S. Gurkaynak, Brian Sack, and Jonathan H. Wright
2006-28

AND

The TIPS Yield Curve and Inflation Compensation
Refet S. Gürkaynak, Brian Sack, and Jonathan H. Wright
2008-05

First, download the data from the following places:

1)  The TIPS Yield Curve and Inflation Compensation
Info: https://www.federalreserve.gov/pubs/feds/2008/200805/200805abs.html
Data: https://www.federalreserve.gov/data/yield-curve-tables/feds200805.csv
https://www.federalreserve.gov/data/yield-curve-tables/feds200805_1.html

2) The U.S. Treasury Yield Curve: 1961 to the Present
Info: https://www.federalreserve.gov/pubs/feds/2006/200628/200628abs.html
Data: https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv
https://www.federalreserve.gov/data/yield-curve-tables/feds200628_1.html

Example
-------

import load_treasury_yields
df = load_treasury_yields.load_nominal_yields(dirpath=DEFAULT_DATA_DIR)

"""

import numpy as np
import pandas as pd
from pathlib import Path


NOMINAL_YIELDS_FILENAME = "feds200628.csv"
TIPS_YIELDS_FILENAME = "feds200805.csv"
DEFAULT_DATA_DIR = Path('../data/')

def load_nominal_yields(dirpath=None):
    """
    Note:
    Series	                    Compounding Convention	      Mnemonic(s)
    ------------------------------------------------------------------------
    Zero-coupon yield	        Continuously Compounded	      SVENYXX
    Par yield	                Coupon-Equivalent	          SVENPYXX
    Instantaneous forward rate	Continuously Compounded	      SVENFXX
    One-year forward rate	    Coupon-Equivalent	          SVEN1FXX
    Parameters	                N/A	                          BETA0 to TAU2

    """
    path = Path(dirpath) / NOMINAL_YIELDS_FILENAME
    df = pd.read_csv(path, skiprows=list(range(9)))
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    return df

def load_tips_yields(dirpath=None):
    """
    NOTES:

    Please note that rates of inflation compensation ("breakeven" rates) incorporate
    inflation risk premiums and the effects of the differential liquidity of TIPS
    and nominal securities. Consequently breakeven rates should not be interpreted
    as estimates of inflation expectations.

    Series	                    Compounding Convention	  Mnemonic(s)
    ------------------------------------------------------------------

    TIPS Yields
    ----------------------
    Zero-coupon	                Continuously Compounded	  TIPSYXX
    Par	                        Coupon-Equivalent	      TIPSPYXX
    Instantaneous forward	    Continuously Compounded	  TIPSFXX
    One-year forward	        Coupon-Equivalent	      TIPS1FXX
    Five-to-ten-year forward	Coupon-Equivalent	      TIPS5F5
    Parameters	                N/A	                      BETA0 to TAU2

    Inflation Compensation
    ----------------------
    Zero-coupon	                Continuously Compounded	  BKEVENXX
    Par	                        Coupon-Equivalent	      BKEVENPYXX
    Instantaneous forward	    Continuously Compounded	  BKEVENFXX
    One-year forward	        Coupon-Equivalent	      BKEVEN1FXX
    Five-to-ten-year forward	Coupon-Equivalent	      BKEVEN5F5

    """
    path = Path(dirpath) / TIPS_YIELDS_FILENAME
    df = pd.read_csv(path, skiprows=list(range(18)))
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    return df

def _demo():
    df_nominal = load_nominal_yields(dirpath=DEFAULT_DATA_DIR)
    df_tips = load_tips_yields(dirpath=DEFAULT_DATA_DIR)
