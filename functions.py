import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks, Input
from statistics import mean
import matplotlib.pyplot as plt
from itertools import combinations

insts = ["AUD", "NZD", "EUR", "GBP", "CAD", "CHF", "JPY", "USD"]
n_insts = len(insts)

def create_currency_pair_list():
    '''Return a list of the 28 currency pairs that comprise our "universe."'''
    currency_pairs = list()
    for c1, c2 in combinations(insts, 2):
        if path.exists('./Daily/' + c1 + c2 +"vector.rds"):
            currency_pairs.append(c1 + c2)
        elif path.exists('./Daily/' + c2 + c1 +"vector.rds"):
            currency_pairs.append(c2 + c1)
    assert len(currency_pairs) == 28
    return currency_pairs

def create_inclusive_array(freq='30Min', year=2019, features=1, C=True,P=True):
    '''Put all currencies and/or all pairs into one dataframe, so that the time
    index exactly aligns: this makes comparisons easier.
    4 features → OHLC
    3 features → HLC
    1 feature → C only.'''
    df = pd.DataFrame()
    if C:
        for i, c1 in enumerate(insts):
            print(c1)
            with open("./"+freq+"_"+str(year)+"/"+c1 +'_'+freq+'.pickle','rb')\
            as pickle_file:
                d = pickle.load(pickle_file).iloc[:,-features:]
            df = pd.merge(df,d, left_index=True,right_index=True, how='outer',\
                    suffixes=(None, c1))
    if P:
        for i, c1 in enumerate(currency_pairs):
            print(c1)
            with open("./"+freq+"_"+str(year)+"/"+c1 +'_'+freq+'.pickle','rb')\
            as pickle_file:
                d = pickle.load(pickle_file).iloc[:,-features:]
            df = pd.merge(df,d, left_index=True, right_index=True,how='outer',\
                    suffixes=(None, c1))
    #assert df.shape[1] == len(insts) * features # 4 columns per currency
    return df.fillna(method="ffill").fillna(method="bfill").dropna()

def split_sequencesXy(data, lookback=5, horizon=1):
    '''split a 3D multivariate sequence into input X and output y'''
    n_inputs = len(data) - lookback - horizon
    #X = np.empty((n_inputs, lookback, data.shape[1]))
    X = np.empty((n_inputs, lookback))
    y = np.empty(n_inputs)
    for i in range(n_inputs):        
        last_obs = i + lookback # last observation for this time step    e.g. 5
        last_prediction = last_obs + horizon # furthest time to predict (not inclusive)    e.g. 6
        X[i] = data[i:last_obs] # HLC                   e.g. [0, 1, 2, 3, 4]    
        y[i] = data[last_prediction] # Next period's C e.g. 5:6 i.e. [5]
    return X, y

def data2Xy(data, lookback=5, horizon=1):
    '''split a 3D multivariate sequence into input X and output y'''
    n_inputs = len(data) - lookback - horizon
    X = np.empty((n_inputs, lookback))
    y = np.empty(n_inputs)
    for i in range(n_inputs):        
        last_obs = i + lookback # last observation for this time step    e.g. 5
        last_prediction = last_obs + horizon # furthest time to predict (not inclusive)    e.g. 6
        X[i, :, 1] = data[i:last_obs] # HLC                   e.g. [0, 1, 2, 3, 4]    
        y[i] = data[last_prediction, -1] # Next period's C e.g. 5:6 i.e. [5]
    return X, y

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

def model_builder2(lstm_units=64, dropout=0.01, channels=1):
    rnn = Sequential([LSTM(units = lstm_units,
        return_sequences = False,  input_shape=(2, 1)),
            Dropout(dropout),
            Dense(channels)]) # Output layer
    learning_rate = 1e-3
    rnn.compile(optimizer = Adam(learning_rate = learning_rate, clipnorm=1.0),
                loss = 'mean_squared_error')
    return rnn

def mape(actual, forecast):
    '''Mean Absolute Percentage Error'''
    #TODO: Line up by index...
    return mean(abs((forecast - actual) / actual))
    #return statistics.mean(abs((forecast - actual) / actual).flatten())

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
            if c12 in currency_pairs:
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
    print(round(sumc))
    print(round(sump))
    #accuracy_diff = np.array(pair_accuracy) - np.array(syn_accuracy)
    #print(mean(accuracy_diff))
    print(f'Took {round(sump)} sec to process pairs but only \
    {round(sumc)} sec to process currencies...')
    shv = "{:.0%}".format((sump-sumc) / sump)
    print('... shaving off ' + shv)

def plot_training(history): # Korstanje [2021]
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

