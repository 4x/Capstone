import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from functions import mape, insts, model_builder
import functions
from sklearn.model_selection import train_test_split
from os import path
import time
from sklearn.preprocessing import MinMaxScaler
import logging
from numpy import empty, squeeze
from tensorflow.keras import callbacks

# parameters
lookback = 1 # number of previous time steps to use as input
n_features = 1
horizon = 5 # number of time steps ahead to predict

# Network hyperparameters
model_path = r'\LSTM_Multivariate.h5'
bch_size = 32

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
    if 'df' not in globals() and 'df' not in locals():
        df = create_inclusive_array()
    train, test = train_test_split(df, shuffle=False)
    unscaled_y = test.iloc[lookback:-horizon, :] # save for later before
    predictions, ctime, ptime = scale_distribute(train, test)
    
    for i in range(predictions.shape[1]):
        mapes.append(mape(unscaled_y.iloc[:, i], predictions[:, i]))
    print_results(ctime, ptime, mapes[:8], mapes[8:])
    divided = divide_currencies(predictions)
    predicted_pairs = predictions[:, 8:]
    true_pairs = np.array(unscaled_y.iloc[:, 8:])
    divided_currency_err = mape(true_pairs, divided)
    market_pair_err = mape(true_pairs, predicted_pairs)
    print(f'Prediction error for market pairs is {mean(market_pair_err)} on average.')
    print(f'But only {mean(divided_currency_err)} if separating to currencies first.')
    improved = 0
    for i in range(pairs):
        print(f'{pairs[i]} direct: {market_pair_err[i]} indirect: {divided_currency_err[i]}')
        if divided_currency_err[i] < market_pair_err[i]: improved += 1
    print(f'Improved results for {improved}/28 currency pairs')
    return predictions, unscaled_y, ctime, ptime, mapes
