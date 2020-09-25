import pandas as pd, pandas_datareader.data as web, datetime as dt, matplotlib.pyplot as plt, numpy as np

def get_data_from_moex (tickers, startdate, enddate):
    df = web.DataReader(tickers, 'moex', startdate, enddate)
    df = df.drop(df.columns[[0, 1, 2, 3, 4, 8, 11, 12, 13, 14, 15, 16, 17, 18]], axis = 1)
    df = df[np.isfinite(df['OPEN'])]
    print(df)
    return(df)

get_data_from_moex('LKOH', '2019-04-01', '2019-05-21')

