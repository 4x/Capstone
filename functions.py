from pandas import DataFrame, merge, concat, Series
import pickle
from numpy import empty, mean, squeeze, column_stack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from keras.regularizers import L1L2
import matplotlib.pyplot as plt
from keras_tuner import Hyperband
from time import perf_counter
import logging
from concurrent.futures import ThreadPoolExecutor
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Global lists
insts = ["AUD", "NZD", "EUR", "GBP", "CAD", "CHF", "JPY", "USD"]
pairs = ['AUDNZD', 'EURAUD', 'GBPAUD', 'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDUSD',\
     'EURNZD', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD', 'EURGBP',\
        'EURCAD', 'EURCHF', 'EURJPY', 'EURUSD', 'GBPCAD', 'GBPCHF', 'GBPJPY',\
        'GBPUSD', 'CADCHF', 'CADJPY', 'USDCAD', 'CHFJPY', 'USDCHF', 'USDJPY']
n_insts = len(insts)
pair_map = [[0, 3, 4, 5, 6], [9, 10, 11, 12], [1, 7, 13, 14, 15, 16, 17],\
    [2, 8, 18, 19, 20, 21], [22, 23], [25], [], [24, 26, 27]]

# parameters
lookback = 1 # number of previous time steps to use as input
horizon = 10 # number of time steps ahead to predict
n_features = 1

# Network hyperparameters
model_path = r'\LSTM_Multivariate.h5'
bch_size = 1

# per hypertuner:
epochs = 1
#learning_rate = 10 ** -5
#lstm_units = 256

def envelope(df):
    '''Splits data into train/test, sends off to scale_distribute for
    fitting and predicting, and measures and reports performance.'''
    mapes = list()
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.WARN, datefmt="%H:%M:%S")
    logger = logging.getLogger('functions')
    if 'df' not in globals() and 'df' not in locals():
        df = create_inclusive_array()
    train, test = train_test_split(df, shuffle=False)
    unscaled_y = test.iloc[lookback:-horizon, :] # save for later before
    predictions, ctime, ptime = scale_distribute(train, test)
    
    for i in range(predictions.shape[1]):
        mapes.append(mape(unscaled_y.iloc[:, i], predictions[:, i]))
    divided = divide_currencies(predictions)
    #predicted_pairs = predictions[:, 8:]
    true_pairs = unscaled_y.iloc[:, 8:]
    divided_currency_err = mape(true_pairs, divided)
    mape_improvement = concat([Series(mapes[8:],
    index=divided_currency_err.index), divided_currency_err], axis=1)\
            .assign(improvement = lambda x: ((x[0] - x[1]) / x[0])*100)
    print_results(ctime, ptime, mapes[:8], mapes[8:], mape_improvement)
    pair = 17 # EUR/USD
    d = DataFrame(column_stack([unscaled_y.iloc[:, 8+pair],
        predictions[:, 8+pair], divided[:, pair]]), index=unscaled_y.index,
        columns=['Actual', 'Direct prediction', 'Divided currencies'])
    plt.plot(d)
    plt.title(pairs[pair])
    plt.show()
    return predictions, divided, unscaled_y, mapes, mape_improvement

def create_inclusive_array(freq='30Min', year=2019, features=1, C=True,P=True):
    '''Put all currencies and/or all pairs into one dataframe, so that the time
    index exactly aligns: this makes comparisons easier.
    4 features → OHLC
    3 features → HLC
    1 feature → C only.'''
    df = DataFrame()
    col = 0
    if C: # include currencies
        for c1 in insts:
            print(c1)
            with open("./"+freq+"_"+str(year)+"/"+c1 +'_'+freq+'.pickle','rb')\
            as pickle_file:
                d = pickle.load(pickle_file).iloc[:,-features:]
            df = merge(df,d, left_index=True,right_index=True, how='outer',\
                    suffixes=(None, c1))
        col += len(insts) * features # 4 columns per currency
    if P: # include pairs
        for c1 in pairs:
            print(c1)
            with open("./"+freq+"_"+str(year)+"/"+c1 +'_'+freq+'.pickle','rb')\
            as pickle_file:
                d = pickle.load(pickle_file).iloc[:,-features:]
            df = merge(df,d, left_index=True, right_index=True,how='outer',\
                    suffixes=(None, c1))
        col += len(pairs) * features # 4 columns per currency
    assert df.shape[1] == col
    return df.fillna(method="ffill").fillna(method="bfill").dropna()

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
    rnn = Sequential([LSTM(units = lstm_units, stateful=False,
    bias_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences = False,
    input_shape=shape),
            Dropout(dropout),
            Dense(channels)]) # Output layer
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

def scale_distribute(train, test):
    '''Scale all prices, distribute fitting and prediction, rescale back.'''
    ptime, ctime = list(), list()
    scaler = MinMaxScaler() # One Scaler for the whole dataframe
    train = scaler.fit_transform(train)
    test = scaler.transform(test) # avoid lookahead bias
    predictions = empty((len(test) - horizon - lookback, 36))
    for i in range(n_insts):
        predictions[:, i], t = pipeline_df(train[:, i], test[:, i])
        ctime.append(t)
        current_pair = pair_map[i]
        with ThreadPoolExecutor() as executor:
            for k, pred_t in enumerate(executor.map(lambda x:
            pipeline_df(train[:, 8+x], test[:, 8+x]), current_pair)):
                if pred_t is not None: # non-currency pairs e.g. USD/GBP
                    predictions[:, 8+current_pair[k]] = pred_t[0]
                    ptime.append(pred_t[1])
                    print(f'{pairs[current_pair[k]]} done')
    predictions = scaler.inverse_transform(predictions) # All at once
    return predictions, ctime, ptime

def pipeline_df(train, test, n_features=1):
    '''All steps necessary from dataframe input to training and prediction.'''
    tic = perf_counter()
    X_train, y_train = splitXy(train, lookback, horizon)
    X_test, _ = splitXy(test, lookback, horizon)
    
    # Construct and train model
    model = model_builder((lookback, 1))
    history = model.fit(X_train, y_train, validation_split=0.2,
        epochs = epochs, batch_size = bch_size, shuffle=False,
        callbacks = [callbacks.EarlyStopping(monitor='val_loss',
        min_delta=0, patience=10, verbose=1, mode='min'),
            callbacks.ModelCheckpoint(model_path, monitor='val_loss',
            save_best_only=True, mode='min', verbose=0)])
    return squeeze(model.predict(X_test)), perf_counter() - tic #,history 

def mape(actual, forecast):
    '''Mean Absolute Percentage Error'''
    return mean(abs((forecast - actual) / actual)) * 100

def map_pairs_to_currency():
    matching_pairs = list()
    for i, c1 in enumerate(insts):
        p = list()
        for j, pair in enumerate(pairs):
            if pair[:3] == c1:
                p.append(j)
        matching_pairs.append(p)
    assert len(matching_pairs) == n_insts
    print(matching_pairs)
    return matching_pairs

def divide_currencies(predictions):
    divided = empty((predictions.shape[0], 28))
    for i, pair in enumerate(pairs):
        base, quote = pair[:3], pair[3:]
        b = insts.index(base)
        q = insts.index(quote)
        divided[:, i] = (predictions[:, b] / predictions[:, q]).flatten()
    return divided

def print_results(ct, pt, currency_accuracy, pair_accuracy, mape_improvement):
    sumc = sum(ct)
    sump = sum(pt)
    print(mape_improvement)
    print(f'Improved results for {mape_improvement[mape_improvement.improvement > 0].count()[0]}/28 currency pairs')
    print('Mean Absolute Percentage Errors:')
    print(f'Pair: {round(mean(pair_accuracy), 2)}%')
    print(f'Currency: {round(mean(currency_accuracy), 2)}% (Lower is better)')
    print(f'Took {round(sump)} sec to process pairs but only \
    {round(sumc)} sec to process currencies...')
    print("... shaving off {:.0%}".format((sump-sumc) / sump))

def plot_training(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training the neural network')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def plot_predictions(x, true_y, direct_prediction, divided_prediction):
    plt.plot(x, true_y)
    plt.plot(x, direct_prediction)
    plt.plot(x, divided_prediction)
    plt.plot(history.history['val_loss'])
    plt.title('Training the neural network')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

# Hyperparameter search/optimization
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
    tuner = Hyperband(model_tuner,
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