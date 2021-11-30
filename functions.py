import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks, Input
from statistics import mean
import matplotlib.pyplot as plt
from os import path

insts = ["AUD", "NZD", "EUR", "GBP", "CAD", "CHF", "JPY", "USD"]
pairs = ['AUDNZD', 'EURAUD', 'GBPAUD', 'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDUSD',\
     'EURNZD', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD', 'EURGBP',\
        'EURCAD', 'EURCHF', 'EURJPY', 'EURUSD', 'GBPCAD', 'GBPCHF', 'GBPJPY',\
        'GBPUSD', 'CADCHF', 'CADJPY', 'USDCAD', 'CHFJPY', 'USDCHF', 'USDJPY']
n_insts = len(insts)

def create_inclusive_array(freq='30Min', year=2019, features=1, C=True,P=True):
    '''Put all currencies and/or all pairs into one dataframe, so that the time
    index exactly aligns: this makes comparisons easier.
    4 features → OHLC
    3 features → HLC
    1 feature → C only.'''
    df = pd.DataFrame()
    col = 0
    if C:
        for c1 in insts:
            print(c1)
            with open("./"+freq+"_"+str(year)+"/"+c1 +'_'+freq+'.pickle','rb')\
            as pickle_file:
                d = pickle.load(pickle_file).iloc[:,-features:]
            df = pd.merge(df,d, left_index=True,right_index=True, how='outer',\
                    suffixes=(None, c1))
        col += len(insts) * features # 4 columns per currency
            if P:
        for c1 in pairs:
            print(c1)
            with open("./"+freq+"_"+str(year)+"/"+c1 +'_'+freq+'.pickle','rb')\
            as pickle_file:
                d = pickle.load(pickle_file).iloc[:,-features:]
            df = pd.merge(df,d, left_index=True, right_index=True,how='outer',\
                    suffixes=(None, c1))
        col += len(pairs) * features # 4 columns per currency
    assert df.shape[1] == col
    return df.fillna(method="ffill").fillna(method="bfill").dropna()

def splitXy(data, lookback=5, horizon=1):
    '''split a 3D multivariate sequence into input X and output y'''
    n_inputs = len(data) - lookback - horizon
    X = np.empty((n_inputs, lookback, 1))
    y = np.empty(n_inputs)
    for i in range(n_inputs):        
        last_obs = i + lookback # last observation for this time step    e.g. 5
        last_prediction = last_obs + horizon # furthest time to predict (not inclusive)    e.g. 6
        X[i] = data[i:last_obs].reshape(-1, 1) # HLC                   e.g. [0, 1, 2, 3, 4]    
        y[i] = data[last_prediction] # Next period's C e.g. 5:6 i.e. [5]
    return X, y

def model_builder(shape, lstm_units=64, dropout=0.01, channels=1):
    rnn = Sequential([LSTM(units = lstm_units,
        return_sequences = False,  input_shape=shape),
        #return_sequences = False),
            Dropout(dropout),
            Dense(channels)]) # Output layer
    learning_rate = 1e-3
    rnn.compile(optimizer = Adam(learning_rate = learning_rate, clipnorm=1.0),
                loss = 'mean_squared_error')
    return rnn

def mape(actual, forecast):
    '''Mean Absolute Percentage Error'''
    return mean(abs((forecast - actual) / actual))

def syn_forecasts():
    tic = time.perf_counter()
    syn_forecast, syn_accuracy, stime = [], [], []
    currency1 = 0
    pair_counter = 0
    for i, c1 in enumerate(insts):
        currency2 = 0
        for j, c2 in enumerate(insts):
            c12 = c1 + c2
            # f = c12 + "vector.rds"
            if c12 in pairs:
                tic = time.perf_counter()
                ff = np.squeeze(currency_frcast[i] / currency_frcast[j])
                m = mape(pair_actual[pair_counter], ff)
                syn_forecast.append(ff)
                syn_accuracy.append(m)
                pair_counter += 1
            currency2 += 1
        currency1 += 1
    print(pair_counter)
    stime.append(time.perf_counter() - tic)
    return syn_forecast, syn_accuracy, stime

def print_results(): # previously in runAll.py
    sumc = sum(ctime)
    sump = sum(ptime)
    print(mean(currency_accuracy))
    print(mean(pair_accuracy))
    #print(mean(syn_accuracy))
    #accuracy_diff = np.array(pair_accuracy) - np.array(syn_accuracy)
    #print(mean(accuracy_diff))
    print(f'Took {round(sump)} sec to process pairs but only \
    {round(sumc)} sec to process currencies...')
    shv = "{:.0%}".format((sump-sumc) / sump)
    print('... shaving off ' + shv)

def plot_training(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training the neural network')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

