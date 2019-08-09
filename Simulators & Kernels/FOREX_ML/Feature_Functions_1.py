import pandas as pd
import numpy as np
from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from decimal import Decimal, ROUND_DOWN, ROUND_UP

class Holder:

    @staticmethod
    def heikenashi(prices, periods):
        """
        <<REMOVES NOISE>>
        :param prices: dataframe of OHLC & Volume data
        :param periods: periods for which to create candles
        :return: hein ken ashi OHLC candles
        """
        results = Holder()
        dict = {}
        HAclose = prices[['open', 'high', 'low', 'close']].sum(axis=1)/4
        HAopen = HAclose.copy()
        HAopen.iloc[0] = HAclose.iloc[0]
        HAhigh = HAclose.copy()
        HAlow = HAclose.copy()
        for i in range(0, len(prices)):
            HAopen.iloc[i] = (HAopen.iloc[i-1] + HAclose.iloc[i-1])/2
            HAhigh.iloc[i] = np.array([prices.high.iloc[i], HAopen.iloc[i], HAclose.iloc[i]]).max()
            HAlow.iloc[i] = np.array([prices.low.iloc[i], HAopen.iloc[i], HAclose.iloc[i]]).min()

        df = pd.concat((HAopen, HAhigh, HAlow, HAclose), axis=1)
        df.columns = [['open', 'high', 'low', 'close']]

        # df.index = df.index.droplevel(0)

        dict[periods[0]] = df

        results.candles = dict
        return results


    @staticmethod
    def detrend(prices, method='difference'):
        """
        :param prices:
        :param method:
        :return:
        """

        if method == 'difference':
            detrended = prices.close[1:] - prices.close[:-1].values

        elif method == 'linear':
            x = np.arange(0, len(prices))
            y = prices.close.values

            model = LinearRegression()

            model.fit(x.reshape(-1,1), y.reshape(-1,1))

            trend = model.predict(x.reshape(-1,1))
            trend = trend.reshape((len(prices),))
            detrended = prices.close - trend

        else:
            print('Call Derrick There is an Error')

        return detrended


    @staticmethod
    def fseries(x, a0,a1, b1, w):
        """
        <<Model Polynomial WITH COSINE>>
        :param x:
        :param a0:
        :param a1:
        :param b1:
        :param w:
        :return:
        """

        f = a0 + a1*np.cos(w*x) + b1*np.sin(w*x)

        return f


    @staticmethod
    def sseries(x, a0, b1, w):
        """
        <<Polynomial Function WITHOUT COSINE>>
        :param x:
        :param a0:

        :param b1:
        :param w:
        :return:
        """

        f = a0 + b1*np.sin(w*x)

        return f


# Fourier Series Coefficient Calculator functions
    @staticmethod
    def fourier(prices, periods, method='difference'):
        """
        <<Models Returns(assuming we use Difference detrend) - Polynomial r/s>>
        :param prices:
        :param periods:
        :param method:
        :return:
        """
        results = Holder()
        dict = {}

        # Option to plot the expansion fit for each iteration
        plot = False

        # Compute the coefficients of the series
        detrended = Holder.detrend(prices, method)
        for i in range(0, len(periods)):
            coeffs = []
            for j in range(periods[i], len(prices)-periods[i]):
                x = np.arange(0, periods[i])
                y = detrended.iloc[j-periods[i]:j]
                with warnings.catch_warnings():
                    warnings.simplefilter('error', OptimizeWarning)
                    try:
                        res = scipy.optimize.curve_fit(Holder.fseries,x,y)
                    except (RuntimeError, OptimizeWarning):
                        res = np.empty((1,4))
                        res[0,:] = np.NAN

                if plot == True:
                    xt = np.linspace(0, periods[i],100)
                    yt = Holder.fseries(xt, res[0][0], res[0][1], res[0][2], res[0][3])
                    plt.plot(x, y)
                    plt.plot(xt, yt, 'r')
                    plt.show()

                coeffs = np.append(coeffs, res[0], axis=0)

            warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

            coeffs = np.array(coeffs).reshape((len(coeffs)/4, 4))

            df = pd.DataFrame(coeffs, index=prices.iloc[periods[i]:-periods[i]])

            df.columns = [['a0', 'a1', 'b1', 'w']]

            df = df.fillna(method='ffill')

            dict[periods[i]] = df

        results.coeffs = dict

        return results


# Sine Series Coefficient Calculator functions
    @staticmethod
    def sine(prices, periods, method='difference'):
        """
        <<Models Returns(assuming we use Difference detrend) - Polynomial r/s>>

        :param prices:
        :param periods:
        :param method:
        :return:
        """
        results = Holder()
        dict = {}

        # Option to plot the expansion fit for each iteration
        plot = False

        # Compute the coefficients of the series
        detrended = Holder.detrend(prices, method)
        for i in range(0, len(periods)):
            coeffs = []
            for j in range(periods[i], len(prices) - periods[i]):
                x = np.arange(0, periods[i])
                y = detrended.iloc[j - periods[i]:j]
                with warnings.catch_warnings():
                    warnings.simplefilter('error', OptimizeWarning)
                    try:
                        res = scipy.optimize.curve_fit(Holder.sseries, x, y)
                    except (RuntimeError, OptimizeWarning):
                        res = np.empty((1, 3))
                        res[0, :] = np.NAN

                if plot == True:
                    xt = np.linspace(0, periods[i], 100)
                    yt = Holder.sseries(xt, res[0][0], res[0][1], res[0][2])
                    plt.plot(x, y)
                    plt.plot(xt, yt, 'r')
                    plt.show()

                coeffs = np.append(coeffs, res[0], axis=0)

            warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

            coeffs = np.array(coeffs).reshape((len(coeffs) / 3, 3))

            df = pd.DataFrame(coeffs, index=prices.iloc[periods[i]:-periods[i]])

            df.columns = [['a0', 'b1', 'w']]

            df = df.fillna(method='ffill')

            dict[periods[i]] = df

        results.coeffs = dict

        return results


# Williams Accumulation Distribution Function
    @staticmethod
    def wadl(prices, periods):
        """
        <<DEMAND & SUPPLY MEASUREMENT>>
        :param prices:
        :param periods:
        :return:
        """
        results = Holder()
        dict = {}

        for i in range(0, len(periods)):
            WAD = []
            for j in range(periods[i], len(prices)-periods[i]):
                TRH = np.array([prices.high.iloc[j], prices.close.iloc[j-1]]).max()
                TRL = np.array([prices.low.iloc[j], prices.close.iloc[j - 1]]).min()
                if prices.close.iloc[j] > prices.close.iloc[j-1]:
                    #PM = Decimal(prices.close.iloc[j] - TRL).quantize(Decimal('1e-4'))
                    PM = float(format(prices.close.iloc[j] - TRL, '.4f'))
                    #PM = prices.close.iloc[j] - TRL
                elif prices.close.iloc[j] < prices.close.iloc[j-1]:
                    #PM = Decimal(prices.close.iloc[j] - TRH).quantize(Decimal('1e-4'))
                    PM = float(format(prices.close.iloc[j] - TRH, '.4f'))
                    #PM = prices.close.iloc[j] - TRH
                if prices.close.iloc[j] == prices.close.iloc[j-1]:
                    PM = 0
                else:
                    # print(prices.close.iloc[j], '   ', prices.close.iloc[j - 1])
                    # print(type(prices.close.iloc[j]), '   ', type(prices.close.iloc[j - 1]))
                    print('Unknown Error Call Derrick')
                AD = PM*prices.volume.iloc[j]
                WAD = np.append(WAD, AD)
            WAD = WAD.cumsum()
            WAD = pd.DataFrame(WAD, index=prices.iloc[periods[i]:-periods[i]].index)
            WAD.columns = [['close']]

            dict[periods[0]] = WAD
        results.wadl = dict
        return results


# Data Re-Sampler
    @staticmethod
    def ohlc_resample(dataFrame, timeFrame, column='ask'):
        """
        :param dataFrame: data-frame containing data that we want to ReSample
        :param timeFrame: time-frame for ReSampling
        :param column: which column to ReSample
        :return: ReSampled OHLC data for the given time-frame
        """
        dataFrame['Symbol'] = 'EURSGD'
        grouped = dataFrame.groupby('Symbol')


        if np.any(dataFrame.columns=='Ask'):
            if column == 'ask':
                ask = grouped['Ask'].resample(timeFrame).ohlc()
                askVol = grouped['AskVol'].resample(timeFrame).count()
                resampled = pd.DataFrame(ask)
                resampled['AskVol'] = askVol
            elif column == 'bid':
                bid = grouped['Bid'].resample(timeFrame).ohlc()
                bidVol = grouped['BidVol'].resample(timeFrame).count()
                resampled = pd.DataFrame(bid)
                resampled['BidVol'] = bidVol
            else:
                print('Check ReSample Data!!!')

        elif np.any(dataFrame.columns == 'close'):
            open = grouped['open'].resample(timeFrame).ohlc()
            high = grouped['high'].resample(timeFrame).ohlc()
            low = grouped['low'].resample(timeFrame).ohlc()
            close = grouped['close'].resample(timeFrame).ohlc()
            askVol = grouped['AskVol'].resample(timeFrame).count()

            resampled = pd.DataFrame(open)
            resampled['high'] = high
            resampled['low'] = low
            resampled['close'] = close
            resampled['AskVol'] = askVol

        resampled = resampled.dropna()
        return resampled


# Momentum
    @staticmethod
    def momentum(prices, periods):
        """
        :param prices: dataframe of OHLC data
        :param periods: lists of periods to calculate function value
        :return: momentum indicator
        """
        results = Holder()
        open = {}
        close = {}

        for i in range(0, len(periods)):
            open[periods[i]] = pd.DataFrame(prices.open.iloc[periods[i]:]-prices.open.iloc[:-periods[i]].values,
                                            index=prices.iloc[periods[i]:].index)
            close[periods[i]] = pd.DataFrame(prices.close.iloc[periods[i]:] - prices.close.iloc[:-periods[i]].values,
                                             index=prices.iloc[periods[i]:].index)
            open[periods[i]].columns = [['open']]
            close[periods[i]].columns = [['close']]
        results.open = open
        results.close = close
        return results


# Stochastic Oscillator Function
    @staticmethod
    def stochastic(prices, periods):
        """
        :param prices: OHLC dataframe
        :param periods: periods to calculate function value
        :return: oscillator function values
        """
        results = Holder()
        close = {}

        for i in range(0, len(periods)):
            Ks = []
            for j in range(periods[i], len(prices)-periods[i]):
                C = prices.close.iloc[j + 1]
                H = prices.high.iloc[j - periods[i]:j].max()
                L = prices.low.iloc[j - periods[i]:j].min()
                if H == L:
                    K = 0
                else:
                    K = 100*(C-L)/(H-L)
                Ks = np.append(Ks, K)

            df = pd.DataFrame(Ks, index=prices.iloc[periods[i] + 1:-periods[i] + 1].index)
            df.columns = [['K']]
            df['D'] = df.K.rolling(3).mean()
            df = df.dropna()

            close[periods[i]] = df

        results.close = close

        return results


# William Oscillator Function
    @staticmethod
    def williams(prices, periods):
        """
        :param prices:
        :param periods:
        :return:
        """
        results = Holder()
        close = {}

        for i in range(0, len(periods)):
            Rs = []
            for j in range(periods[i], len(prices)-periods[i]):
                C = prices.close.iloc[j + 1]
                H = prices.high.iloc[j - periods[i]:j].max()
                L = prices.low.iloc[j - periods[i]:j].min()
                if H == L:
                    R = 0
                else:
                    R = -100*(H-C)/(H-L)
                Rs = np.append(Rs, R)

            df = pd.DataFrame(Rs, index=prices.iloc[periods[i] + 1:-periods[i] + 1].index)
            df.columns = [['R']]
            df = df.dropna()

            close[periods[i]] = df

        results.close = close

        return results


# PROC function (Price Rate of Change) i.e.Slope or perc% change from one value to another
    @staticmethod
    def proc(prices, periods):
        """
        :param prices: dataframe containing prices
        :param periods: periods for which to calculate PROC
        :return: PROC for periods indicated
        """
        results = Holder()
        proc = {}

        for i in range(0, len(periods)):
            proc[periods[i]] = pd.DataFrame(prices.close.iloc[periods[i]:]-prices.close.iloc[:-periods[i]].values / prices.close.iloc[:-periods[i]].values)
            proc[periods[i]].columns = [['close']]

        results.proc = proc

        return results


# Accumulation Distribution Oscillator i.e.Similar to WADL????
    @staticmethod
    def adosc(prices, periods):
        """
        :param prices: OHLC dataframe
        :param periods: periods for which to compute indicator
        :return: indicator values for indicated periods
        """
        results = Holder()
        accdist = {}

        for i in range(0, len(periods)):
            AD = []
            for j in range(periods[i], len(prices)-periods[i]):
                C = prices.close.iloc[j + 1]
                H = prices.high.iloc[j - periods[i]:j].max()
                L = prices.low.iloc[j - periods[i]:j].min()
                V = prices.AskVol.iloc[j + 1]
                if H == L:
                    CLV = 0
                else:
                    CLV = ((C-L)-(H-C))/(H-L)
                AD = np.append(AD, CLV*V)

            AD = AD.cumsum()
            AD = pd.DataFrame(AD, index=prices.iloc[periods[i] + 1:-periods[i] + 1].index)
            AD.columns = [['AD']]

            accdist[periods[i]] = AD

        results.AD = accdist

        return results


# MACD Moving Average Convergence Divergence
    @staticmethod
    def macd(prices, periods):
        """
        :param prices: OHLC dataframe of prices
        :param periods: 1x2 Array containing values for the EMAs
        :return: MACD for given periods
        """
        results = Holder()

        EMA_1 = prices.close.ewm(span=periods[0]).mean()
        EMA_2 = prices.close.ewm(span=periods[1]).mean()

        MACD = pd.DataFrame(EMA_1 - EMA_2)
        MACD.columns = [['L']]

        SigMACD = MACD.rolling(3).mean()
        SigMACD.columns = [['SL']]

        results.line = MACD
        results.signal = SigMACD

        return results


# CCI Commodity Channel index i.e.https://www.youtube.com/watch?v=IJ9I_5M6SUM
    @staticmethod
    def cci(prices, periods):
        """
        :param prices:
        :param periods:
        :return:
        """
        results = Holder()
        CCI = {}

        # https://blog.quantinsti.com/build-technical-indicators-in-python/
        for i in range(0, len(periods)):
            TP = (prices.high + prices.low + prices.close) / 3
            MA = prices.close.rolling(periods[i]).mean()
            std = prices.close.rolling(periods[i]).std()

            CCI[periods[i]] = pd.DataFrame((TP - MA) / (0.015 * std))
            CCI[periods[i]].columns = [['close']]

        results.CCI = CCI

        return results


# Bollinger Bands
    @staticmethod
    def bollinger(prices, periods, deviations):
        """
        :param prices:
        :param periods:
        :param deviations:
        :return:
        """
        results = Holder()
        boll = {}

        for i in range(0, len(periods)):
            mid = prices.close.rolling(periods[i]).mean()
            std = prices.close.rolling(periods[i]).std()

            upper = mid + deviations * std
            lower = mid - deviations * std

            df = pd.concat((upper, mid, lower), axis=1)
            df.columns = [['upper', 'mid', 'lower']]

            boll[periods[i]] = df

        results.bands = boll

        return results


# Price Averages i.e.Rolling Mean
    @staticmethod
    def paverage(prices, periods):
        """
        :param prices: OHLC data
        :param periods: list of periods for which to calculate indicator values
        :return: averages over the given period
        """
        results = Holder()
        avgs = {}
        for i in range(0, len(periods)):
            avgs[periods[i]] = pd.DataFrame(prices[['open', 'high', 'low', 'close']].rolling(periods[i]).mean())

        results.avgs = avgs
        return results


# Slope Function i.e.Fit LinearRegressionFunction into specified periods
    @staticmethod
    def slopes(prices, periods):
        """
        :param prices: OHLC data
        :param periods: periods to get indicator values
        :return: slopes over given periods
        """
        results = Holder()
        slope = {}
        for i in range(0, len(periods)):
            ms = []
            for j in range(periods[i], len(prices) - periods[i]):
                y = prices.high.iloc[j-periods[i]:j].values
                x = np.arange(0, len(y))

                res = stats.linregress(x=x, y=y)
                m = res.slope

                ms = np.append(ms, m)

            ms = pd.DataFrame(ms, index=prices.iloc[periods[i] + 1:-periods[i] + 1].index)

            ms.columns = [['high']]

            slope[periods[i]] = ms

        results.slope = slope
        return results






















