from keras_tuner import Hyperband
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from keras.regularizers import L1L2
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

def model_tuner(hp):
    channels=1
    shape = (lookback, 1)
    rnn = Sequential()
    hp_units = hp.Int('units', min_value=1, max_value=101, step=25)
    rnn.add(LSTM(units=hp_units, stateful=False,
    bias_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences = False,
        input_shape=shape))
    rnn.add(Dropout(dropout))
    rnn.add(Dense(channels))
    hp_learning_rate = hp.Choice('learning_rate', values=[3e-5, 1e-4, 1e-3])
    rnn.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                loss='mean_squared_error', metrics=['accuracy'])
    return rnn

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

def plot_training(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training the neural network')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
