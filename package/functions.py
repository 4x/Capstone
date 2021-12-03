from pandas import DataFrame, merge, concat, Series
from numpy import empty, mean, squeeze, column_stack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from keras.regularizers import L1L2
from time import perf_counter
import logging
from concurrent.futures import ThreadPoolExecutor
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# parameters
lookback = 4 # number of previous time steps to use as input
horizon = 4 # number of time steps ahead to predict
n_features = 1

# Network hyperparameters
model_path = '\LSTM_Multivariate.h5'
bch_size = 10
dropout = 1e-4

# per hypertuner:
epochs = 1
learning_rate = 1e-3
lstm_units = 32

def envelope(df=None):
    '''Splits data into train/test, sends off to scale_distribute for
    fitting and predicting, and measures and reports performance.'''
    
    # pre-processing
    mapes = list()
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.WARN, datefmt="%H:%M:%S")
    logger = logging.getLogger('functions')
    if ('df' not in globals() and 'df' not in locals()) or df is None:
        df = create_inclusive_array()
    train, test = train_test_split(df, shuffle=False)
    unscaled_y = test.iloc[lookback:-horizon, :] # save for later before
    
    # fit and predict
    predictions, ctime, ptime = scale_distribute(train, test)
    
    # post processing
    return(unscaled_y, predictions)
    plot_random(unscaled_y, predictions)
    for i in range(predictions.shape[1]):
        mapes.append(mape(unscaled_y.iloc[:, i], predictions[:, i]))
    divided = divide_currencies(predictions)
    divided_currency_err = mape(unscaled_y.iloc[:, 8:], divided)
    mape_improvement = concat([Series(mapes[8:],
    index=divided_currency_err.index), divided_currency_err], axis=1)\
            .assign(improvement = lambda x: ((x[0] - x[1]) / x[0]) * 100)
    print_results(ctime, ptime, mapes[:8], mapes[8:], mape_improvement)
    plot_random(unscaled_y, predictions, divided)
    return predictions, divided, unscaled_y, mapes, mape_improvement

def splitXy(data, lookback=5, horizon=1):
    '''split a 3D multivariate sequence into input X and output y'''
    n_inputs = len(data) - lookback - horizon
    X, y = empty((n_inputs, lookback, 1)), empty(n_inputs)
    for i in range(n_inputs):        
        last_obs = i + lookback # last observation for this time step    e.g. 5
        last_prediction = last_obs + horizon # furthest time to predict (not inclusive)    e.g. 6
        X[i] = data[i:last_obs].reshape(-1, 1) # HLC                   e.g. [0, 1, 2, 3, 4]    
        y[i] = data[last_prediction] # Next period's C e.g. 5:6 i.e. [5]
    return X, y

def model_builder(shape, lstm_units=256, dropout=0.01, channels=1):
    rnn = Sequential()
    rnn.add(LSTM(units = lstm_units, stateful=False,
    bias_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences = False,
    input_shape=shape))
    rnn.add(Dropout(dropout))
    rnn.add(Dense(channels))
    rnn.compile(optimizer = Adam(learning_rate = learning_rate),
                loss='mean_squared_error', metrics=['accuracy'])
    return rnn

def scale_distribute(train, test):
    '''Scale all prices, distribute fitting and prediction, rescale back.'''
    ptime, ctime = list(), list()
    scaler = MinMaxScaler() # One Scaler for the whole dataframe
    train = scaler.fit_transform(train)
    test = scaler.transform(test) # avoid lookahead bias
    predictions = empty((len(test) - horizon - lookback, test.shape[1]))
    #for i in range(len(insts)):
    for i, current_pair in enumerate(pair_map):
        predictions[:, i], t = pipeline_df(train[:, i], test[:, i])
        ctime.append(t)
        with ThreadPoolExecutor() as executor:
            for k, pred_t in enumerate(executor.map(lambda x:
            pipeline_df(train[:, 8+x], test[:, 8+x]), current_pair)):
                if pred_t is not None: # non-currency pairs e.g. USD/GBP
                    predictions[:, 8+current_pair[k]] = pred_t[0]
                    ptime.append(pred_t[1])
                    print(f'{pairs[current_pair[k]]} done')
    #predictions = scaler.inverse_transform(predictions) # All at once
    #return predictions, ctime, ptime
    return scaler.inverse_transform(predictions), ctime, ptime

def pipeline_df(train, test, n_features=1):
    '''All steps necessary from dataframe input to training and prediction.'''
    tic = perf_counter()
    X_train, y_train = splitXy(train, lookback, horizon)
    X_test, _ = splitXy(test, lookback, horizon)
    
    # Construct and train model
    model = model_builder((lookback, 1), lstm_units, dropout)
    history = model.fit(X_train, y_train, validation_split=0.2,
        epochs = epochs, batch_size = bch_size, shuffle=False,
        callbacks = [callbacks.EarlyStopping(monitor='val_loss',
        min_delta=0, patience=10, verbose=1, mode='min'),
            callbacks.ModelCheckpoint(model_path, monitor='val_loss',
            save_best_only=True, mode='min', verbose=0)])
    #return model.predict(X_test), perf_counter() - tic #,history
    return squeeze(model.predict(X_test)), perf_counter() - tic #,history
