import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from functions import mape, insts, split_sequencesXy, model_builder
import functions
from sklearn.model_selection import train_test_split
from os import path
import time
from sklearn.preprocessing import MinMaxScaler
import logging
import concurrent.futures
import numpy as np
from tensorflow.keras import callbacks

# parameters
lookback = 5 # number of previous time steps to use as input
n_features = 1
horizon = 5 # number of time steps ahead to predict

# Network hyperparameters
model_path = r'\LSTM_Multivariate.h5'
epochs = 5 # Loss appears to flatten around 80
bch_size = 32
learning_rate = 0.01
lstm_units = 64

def distribute_predictions(prefix='./30Min_2019/', suffix = '_30Min.pickle'):
    ptime, ctime = list(), list()
    pair_frcast, currency_frcast = list(), list()
    currency_accuracy, pair_accuracy = list(), list()
    currency_actual, syn_actual, pair_actual = list(), list(), list()
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.WARN, datefmt="%H:%M:%S")
    df = create_inclusive_array()
    train, test = train_test_split(df, shuffle=False)
    unscaled_y = test.iloc[lookback:-horizon, :]
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test) # avoid lookahead bias
    for i, c1 in enumerate(insts):
        a, p, y, t = pipeline_df(train[:, i], test[:, i])
        currency_accuracy.append(a)
        currency_frcast.append(p)
        currency_actual.append(y)
        ctime.append(t)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for _ in executor.map(lambda x:
            pipeline_df(train.oc[:, 8+x], test[:, 8+x]), range(28)):
                if _ is not None:
                    pair_accuracy.append(_[0])
                    pair_frcast.append(_[1])
                    pair_actual.append(_[2])
                    ptime.append(_[3])
    return currency_accuracy, currency_frcast, currency_actual, ctime,\
        pair_accuracy, pair_frcast, pair_actual, ptime

#def pipeline_df(df, n_features=1):
def pipeline_df(train, test, n_features=1):
    '''All steps necessary from dataframe input to training and prediction.'''
    # # Split and scale
    # train, test = train_test_split(df, shuffle=False)
    # unscaled_y = test.iloc[lookback:-horizon]
    # # unscaled_y = test.iloc[lookback:-horizon, -1]
    # scaler = MinMaxScaler()
    # train = scaler.fit_transform(train.reshape(-1, 1))
    # test = scaler.transform(test.reshape(-1, 1)) # avoid lookahead bias
    X_train, y_train = splitXy(train,lookback, horizon)
    X_test, y_test = splitXy(test, lookback, horizon)
    
    # Construct and train model
    model = model_builder2()
    history = model.fit(X_train, y_train, validation_split=0.2,
    epochs = epochs, batch_size = bch_size,\
        callbacks = [callbacks.EarlyStopping(monitor='val_loss',\
        min_delta=0,patience=10,verbose=1,mode='min'),\
            callbacks.ModelCheckpoint(model_path,monitor='val_loss',\
            save_best_only=True, mode='min', verbose=0)])

    # Predict, evaluate, and report
    prediction = scaler.inverse_transform(model.predict(X_test))
    model.save(f.replace('.pickle', '_Model'))
    print(f.replace('./30Min_2019/', '').replace('_30Min.pickle', ''))
    return mape(unscaled_y, np.squeeze(prediction)), prediction,\
        unscaled_y, time.perf_counter() - tic
