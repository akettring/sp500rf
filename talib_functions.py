
# http://mrjbq7.github.io/ta-lib/

from talib import abstract
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt



# get the stock stats
def get_stats(df_in):
    # set up for talib
    df = df_in.copy()
    da = {
        'open': df_in['Open'],
        'high': df_in['High'],
        'low': df_in['Low'],
        'close': df_in['Close'],
        'volume': df_in['Volume']
    }

    # run talib and build df
    # Overlap
    df['SMA_10'] = abstract.SMA(da, timeperiod=10)
    df['SMA_50'] = abstract.SMA(da, timeperiod=50)
    df['SMA_100'] = abstract.SMA(da, timeperiod=100)
    df['SMA_200'] = abstract.SMA(da, timeperiod=200)
    df['MAMA'], df['FAMA'] = abstract.MAMA(da)
    df['SAR'] = abstract.SAR(da)
    # Momentum
    df['ADX'] = abstract.ADX(da)
    df['ADXR'] = abstract.ADXR(da)
    df['AROONOSC'] = abstract.AROONOSC(da)
    df['BOP'] = abstract.BOP(da)
    df['CCI'] = abstract.CCI(da)
    df['CMO'] = abstract.CMO(da)
    df['DX'] = abstract.DX(da)
    macd, macdsignal, df['MACD'] = abstract.MACD(da)
    df['MFI'] = abstract.MFI(da)
    df['MINUS_DI'] = abstract.MINUS_DI(da)
    df['PLUS_DI'] = abstract.PLUS_DI(da)
    df['PPO'] = abstract.PPO(da)
    df['ROCP'] = abstract.ROCP(da)
    df['RSI'] = abstract.RSI(da)
    df['RSI6'] = abstract.RSI(da, timeperiod=6)
    df['RSI100'] = abstract.RSI(da, timeperiod=100)
    slowk, df['slowd'] = abstract.STOCH(da)
    df['fastk'], df['fastd'] = abstract.STOCHF(da)
    df['STOCHRSIk'], df['STOCHRSId'] = abstract.STOCHRSI(da)
    df['TRIX'] = abstract.TRIX(da)
    df['ULTOSC'] = abstract.ULTOSC(da)
    df['WILLR'] = abstract.WILLR(da)
    # Cycle
    df['HT_TRENDLINE'] = abstract.HT_TRENDLINE(da)
    df['HT_DCPERIOD'] = abstract.HT_DCPERIOD(da)
    df['HT_DCPHASE'] = abstract.HT_DCPHASE(da)
    df['HT_inphase'], df['HT_quadrature'] = abstract.HT_PHASOR(da)
    df['HT_sine'], df['HT_leadsine'] = abstract.HT_SINE(da)
    df['HT_TRENDMODE'] = abstract.HT_TRENDMODE(da)
    # Statistics
    df['BETA'] = abstract.BETA(da)
    df['CORREL'] = abstract.CORREL(da)
    df['LINEARREG'] = abstract.LINEARREG(da)
    df['LINEARREG_ANGLE'] = abstract.LINEARREG_ANGLE(da)
    df['LINEARREG_INTERCEPT'] = abstract.LINEARREG_INTERCEPT(da)
    df['LINEARREG_SLOPE'] = abstract.LINEARREG_SLOPE(da)
    df['STDDEV'] = abstract.STDDEV(da)
    df['TSF'] = abstract.TSF(da)

    return(df)





# SCALE FEATURE VALUES BASED ON CORRELATION
def scale_vals_corr(df, cors1=None, cors2=None):
    print("scaling features...")

    # get correlations to close vals
    cors = abs(df.drop('Ticker', axis=1)).corr().sort_values('Close', ascending=False)['Close']

    # split high and moderate correlation
    if (cors1 == None or cors2 == None):
        cors1 = cors[cors > 0.8].index.to_list()
        cors2 = cors[(0.4 < cors) & (cors < 0.8)].index.to_list()

    # scale HIGH correlation as percent change
    close_orig = df.copy()['Close']
    # new_val = close - old_val
    df[cors1] = df[cors1].apply(lambda row: close_orig - row, axis=0)
    # new_val = old_val / close
    df[cors1] = df[cors1].apply(lambda row: row / close_orig, axis=0)
    df[cors1] = df[cors1] * 100

    # scale MODERATE correlation by simple division
    # new_val = old_val / close
    df[cors2] = df[cors2].apply(lambda row: row / close_orig, axis=0)

    return df, cors1, cors2





# auto scale values (per batch)
def scale_vals_auto(df_in, scaler=None):
    df=df_in.copy()

    # preprocess numpy array with sklearn (preserve index)
    rn = df.index
    cn = df.columns
    x = df.values

    # if no scaler is supplied, make one
    if scaler is None:
        scaler = preprocessing.StandardScaler()
        x_scaled = scaler.fit_transform(x)
    # otherwise use the supplied scaler
    else:
        x_scaled = scaler.transform(x)

    # add the index back
    df = pd.DataFrame(x_scaled)
    df.columns = cn
    df.index = rn


    return df, scaler


