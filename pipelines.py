import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from functions import mape, insts, model_builder
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
lookback = 1 # number of previous time steps to use as input
n_features = 1
horizon = 5 # number of time steps ahead to predict

# Network hyperparameters
model_path = r'\LSTM_Multivariate.h5'
epochs = 5 # Loss appears to flatten around 80
bch_size = 32
learning_rate = 0.001
lstm_units = 32

def distribute_predictions(df):
    ptime, ctime = list(), list()
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.WARN, datefmt="%H:%M:%S")
    #if 'df' not in globals: df = create_inclusive_array()
    train, test = train_test_split(df, shuffle=False)
    unscaled_y = test.iloc[lookback:-horizon, :]
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test) # avoid lookahead bias
    n_predictions = len(test) - horizon - lookback
    predictions = np.empty((n_predictions, 36))
    for i, _ in enumerate(insts):
        predictions[:, i], t = pipeline_df(train[:, i], test[:, i])
        ctime.append(t)
        current_pair = pair_map[i]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for k, pred_t in enumerate(executor.map(lambda x:
            pipeline_df(train[:, 8+x], test[:, 8+x]), current_pair)):
                if pred_t is not None: # non-currency pairs e.g. USD/GBP
                    predictions[:, 8+current_pair[k]] = pred_t[0]
                    ptime.append(pred_t[1])
                    print(f'{pairs[current_pair[k]]} done')
    predictions = scaler.inverse_transform(predictions)
    #return predictions, unscaled_y-predictions, mape(unscaled_y, predictions)
    mapes = list()
    for i in range(predictions.shape[1]):
        mapes.append(mape(unscaled_y.iloc[:, i], predictions[:, i]))
    print_results(ctime, ptime, mapes[:8], mapes[8:])
    return predictions, unscaled_y, ctime, ptime, mapes

def pipeline_df(train, test, n_features=1):
    '''All steps necessary from dataframe input to training and prediction.'''
    tic = time.perf_counter()
    X_train, y_train = splitXy(train,lookback, horizon)
    X_test, y_test = splitXy(test, lookback, horizon)
    
    # Construct and train model
    model = model_builder((lookback, 1))
    history = model.fit(X_train, y_train, validation_split=0.2,
    # epochs = epochs, batch_size = bch_size,\
    epochs = epochs,\
        callbacks = [callbacks.EarlyStopping(monitor='val_loss',\
        min_delta=0,patience=10,verbose=1,mode='min'),\
            callbacks.ModelCheckpoint(model_path,monitor='val_loss',\
            save_best_only=True, mode='min', verbose=0)])
    return np.squeeze(model.predict(X_test)), time.perf_counter() - tic
