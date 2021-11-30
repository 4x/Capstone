import pandas as pd
import pickle
from os import path
from utilities import create_currency_pair_list
from functions import insts

def create_ticks(year=2019):
    ''' Create currency ticks from Dukascopy/JForex yearly csv files'''
    #TODO create ./Ticks_year directory if does not exist
    suffix = "_Ticks_" + year + ".01.01_" + year + ".12.31.csv"
    f1 = "DOLLARIDXUSD" + suffix
    usdf = pd.read_csv(f1, index_col=0, parse_dates=True,
                       infer_datetime_format=True, usecols=[0, 1])
    for c1 in insts:
        f = c1 + "USD" + suffix
        if path.exists(f):
            print(c1)
            d = pd.merge(pd.read_csv(f, index_col=0, parse_dates=True,
            infer_datetime_format=True, usecols=[0, 1]), usdf, left_index=True,
                right_index=True, how="outer").fillna(method="ffill").dropna()

            d['Ask'] = d['Ask'] * d['USD']
            d.drop(['USD'], axis=1, inplace=True)

            with open(c1, "wb") as pickle_file:
                pickle.dump("./Ticks_" + year + "/" + d, pickle_file)
        else:
            f = "USD" + c1 + suffix
            if path.exists(f):
                print(c1)
                d = pd.merge(pd.read_csv(f, index_col=0, parse_dates=True,
                infer_datetime_format=True, usecols=[0, 1]), usdf,
                left_index=True, right_index=True, how="outer"). \
                    fillna(method="ffill").dropna()

                d['Ask'] = d['USD'] / d['Ask']  # e.g. USD / USD/CAD
                d.drop(['USD'], axis=1, inplace=True)

                with open(c1, "wb") as pickle_file:
                    pickle.dump("./Ticks_" + year + "/" + d, pickle_file)

def create_ts(freq='1H'):
    '''Resample tick data for required frequency OHLC.'''
    currency_pairs = create_currency_pair_list()
    c_len, p_len = list(), list() # lengths of resulting ts
    for i, c1 in enumerate(insts):
        print(c1)
        with open(c1 + 'ticks.pickle', 'rb') as pickle_file:
            base = pickle.load(pickle_file)
        p = base.resample(freq).ohlc()
        ohlc_columns = p.columns.get_level_values(1)
        p.columns = ohlc_columns
        c_len.append(len(p))
        with open(c1 + "_" + freq + ".pickle", "wb") as pickle_file:
            pickle.dump(p, pickle_file)
        for c2 in insts:
            c12 = c1 + c2
            if c12 in currency_pairs:
                print(c12)
                with open(c2 + 'ticks.pickle', 'rb') as pickle_file:
                    quote = pickle.load(pickle_file)
                p = pd.merge(base, quote, left_index=True, right_index=True,
                    how="outer").fillna(method="ffill").dropna()
                p['Ask'] = p['Ask_x'] / p['Ask_y']
                p.drop(['Ask_x', 'Ask_y'], axis=1, inplace=True)
                p = p.resample(freq).ohlc()
                p_len.append(len(p))
                p.columns = ohlc_columns
                with open(c12 + "_" + freq + ".pickle", "wb") as pickle_file:
                    pickle.dump(p, pickle_file)
    assert(len(c_len) == 8)
    assert(len(p_len) == 28)
        
