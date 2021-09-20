
# http://mrjbq7.github.io/ta-lib/


import yfinance as yf
import random
from talib import abstract
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


################################################
# FINANCE FUNCTIONS
################################################


# GET SP500 DATA (OR RANDOM SUBSAMPLE) FROM YFINANCE
def get_stock_data(subsample=None, period='2y'):
    # There are 2 tables on Wikipedia we want the first table
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    # download the data, subsample if desired
    symbols = sp500['Symbol'].to_list()
    if subsample:
        symbols = random.sample(symbols, subsample)
    data = yf.download(symbols, period=period)
    # flatten the data, the sort
    data = data.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    data = data.sort_values(by=['Ticker', 'Date'])
    return symbols, data



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


# SPLIT DATA FOR TIMESERIES
def split_data_timeseries(stats=None, split=0.8):
    # before split -> train, after split -> test
    train_len = round( len(stats) * split )
    train_dat = stats[0:train_len -1]
    test_dat = stats[train_len:]
    # all but last column are features (X)
    trainX = train_dat.iloc[:,:-1]
    testX = test_dat.iloc[:, :-1]
    # last column is target (Y)
    trainY = train_dat.iloc[:, -1]
    testY = test_dat.iloc[:, -1]
    return trainX, testX, trainY, testY



# ITERATE AND ADD FEATURES
def add_features(symbols, data):
    print("adding features...")

    # BLANK LISTS TO FILL WITH DFs
    trainX_list = list()
    testX_list = list()
    trainY_list = list()
    testY_list = list()

    # iterate symbols
    for sym in symbols:

        try:
            # FEATURE and TARGET ENGINEERING
            # get the subframe for the symbol
            subframe = data[data['Ticker']==sym]

            # add features, technical indicators from TA-lib
            stats = get_stats(subframe)
            # add target col, tomorrow's price today!
            stats['Close_futr'] = stats['Close'].pct_change().shift(-1)

            # keep complete cases only
            stats = stats.dropna()
            # split the data
            trainX, testX, trainY, testY = split_data_timeseries(stats=stats, split=0.8)

            # DATA MERGE DFs to LISTS
            trainX_list.append(trainX)
            testX_list.append(testX)
            trainY_list.append(trainY)
            testY_list.append(testY)

        except:
            print("Failed:", sym)

    # DATA MERGE LISTS to FINAL DFs
    trainXdf = pd.concat(trainX_list)
    testXdf = pd.concat(testX_list)
    trainYdf = pd.concat(trainY_list)
    testYdf = pd.concat(testY_list)

    return (trainXdf, testXdf, trainYdf, testYdf)


################################################
# FEATURE SCALING FUNCTIONS
################################################

# SCALE VALUES BASED ON CORRELATION
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


# SCALE VALUES USING SKLEARN
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



################################################
# TARGET BOOLEAN FUNCTIONS
################################################


# +/- to boolean
def target_up_dn(trainY, testY):
    trainY = np.sign(trainY).replace(-1, 0)
    testY = np.sign(testY).replace(-1, 0)
    return trainY, testY


# z score to boolean based on threshold for z
def target_zscore(trainY, testY, threshold=0.4):
    # use the train values to transform test values
    meany = trainY.mean()
    stdy = np.std(trainY)
    # calcaulte Z
    trainY = (trainY - meany) / stdy
    # convert to boolean
    trainY[np.abs(trainY)>threshold] = 1
    trainY[np.abs(trainY)<threshold] = 0
    # repeat for test
    testY = (testY - meany) / stdy
    testY[np.abs(testY)>threshold] = 1
    testY[np.abs(testY)<threshold] = 0
    return trainY, testY
