import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks, Input, losses
from statistics import mean
import matplotlib.pyplot as plt
from os import path
import keras_tuner

insts = ["AUD", "NZD", "EUR", "GBP", "CAD", "CHF", "JPY", "USD"]
pairs = ['AUDNZD', 'EURAUD', 'GBPAUD', 'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDUSD',\
     'EURNZD', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD', 'EURGBP',\
        'EURCAD', 'EURCHF', 'EURJPY', 'EURUSD', 'GBPCAD', 'GBPCHF', 'GBPJPY',\
        'GBPUSD', 'CADCHF', 'CADJPY', 'USDCAD', 'CHFJPY', 'USDCHF', 'USDJPY']
n_insts = len(insts)
pair_map = [[0, 3, 4, 5, 6], [9, 10, 11, 12], [1, 7, 13, 14, 15, 16, 17],\
    [2, 8, 18, 19, 20, 21], [22, 23], [25], [], [24, 26, 27]]

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

def model_builder(shape, lstm_units=256, dropout=0.01, channels=1):
    rnn = Sequential([LSTM(units = lstm_units,
        return_sequences = False,  input_shape=shape),
            Dropout(dropout),
            Dense(channels)]) # Output layer
    #learning_rate = 10 ** -5
    rnn.compile(optimizer = Adam(learning_rate = 10 ** -5),
                loss = 'mean_squared_error')
    return rnn

def model_tuner(hp):
    dropout=0.01
    channels=1
    shape = (lookback, 1)
    rnn = Sequential()
    hp_units = hp.Int('units', min_value=256, max_value=512, step=16)
    rnn.add(LSTM(units=hp_units, return_sequences = False,\
        input_shape=shape))
    rnn.add(Dropout(dropout))
    rnn.add(Dense(channels))
    hp_learning_rate = hp.Choice('learning_rate', values=[3e-1, 1e-1, 1e-2])
    rnn.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                loss='mean_squared_error', metrics=['accuracy'])
    return rnn

def scale_fit_predict(train, test):
    ptime, ctime = list(), list()
    scaler = MinMaxScaler() # One Scaler for the whole dataframe
    train = scaler.fit_transform(train)
    test = scaler.transform(test) # avoid lookahead bias
    predictions = empty((len(test) - horizon - lookback, 36))
    for i in range(n_insts):
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
    predictions = scaler.inverse_transform(predictions) # All at once
    return predictions, ctime, ptime

def mape(actual, forecast):
    '''Mean Absolute Percentage Error'''
    #if actual.ndim > 1: actual = np.reshape(actual, -1)
    #if forecast.ndim > 1: forecast = np.reshape(forecast, -1)
    return np.mean(abs((forecast - actual) / actual))

def map_pairs_to_currency():
    matching_pairs = list()
    for i, c1 in enumerate(insts):
        _ = list()
        for j, pair in enumerate(pairs):
            if pair[:3] == c1:
                _.append(j)
        matching_pairs.append(_)
    assert len(matching_pairs) == n_insts
    print(matching_pairs)
    return matching_pairs

def divide_currencies(predictions):
    divided = np.empty((predictions.shape[0], 28))
    for i, pair in enumerate(pairs):
        base, quote = pair[:3], pair[3:]
        b = insts.index(base)
        q = insts.index(quote)
        divided[:, i] = (predictions[:, b] / predictions[:, q]).flatten()
    return divided

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

def print_results(ctime, ptime, currency_accuracy, pair_accuracy):
    sumc = sum(ctime)
    sump = sum(ptime)
    print('Mean Absolute Percentage Error')
    print(f'Pair MAPE: {mean(pair_accuracy)}')
    print(f'Currency MAPE: {mean(currency_accuracy)} (Lower is desirable)')
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

def prepare_run_hypersearch(df, col=0):
    train, test = train_test_split(df, shuffle=False)
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)[:, col]
    test = scaler.transform(test)[:, col]
    X_train, y_train = splitXy(train,lookback, horizon)
    X_test, y_test = splitXy(test, lookback, horizon)
    return optimize_nn(X_train, y_train, X_test, y_test)

def optimize_nn(X, y, X_test, y_test):
    stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner = keras_tuner.Hyperband(model_tuner,
                        objective='val_accuracy',
                        max_epochs=10,
                        factor=3,
                        directory='Hypertuning',
                        project_name='Capstone')
    
    tuner.search(X, y, epochs=50, validation_split=0.2, callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    units = best_hps.get('units')
    learning_rate = best_hps.get('learning_rate')
    print(f"""
    The optimal number of units in the first densely-connected layer is
    {units} and the optimal learning rate for the optimizer is {learning_rate}.
    """)
    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X, y, epochs=25, validation_split=0.2)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))    
    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(X, y, epochs=best_epoch, validation_split=0.2)
    eval_result = hypermodel.evaluate(X_test, y_test)
    print("[test loss, test accuracy]:", eval_result)
    return units, learning_rate, best_epoch