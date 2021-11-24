import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dropout, Dense
from keras.callbacks import EarlyStopping
from keras import optimizers
from kerastuner import HyperModel
from kerastuner.tuners import Hyperband
import os
import sys
from itertools import islice
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)
head, tail = os.path.split(dirname)
head, tail = os.path.split(head)

TIME_STEPS = 90
SCROLLING = 10
NUM_FEATURES = 6

def split_X_y(list):
    X = []
    y = []

    for el in list:
        X.append(el[:-1])
        y.append(el[-1])

    return X, y

def sliding_windows(iterable, n = TIME_STEPS, scrolling = SCROLLING):if scrolling == 0: # otherwise infinte loop
        raise ValueError("Parameter 'scrolling' can't be 0")
    lst = list(iterable)
    i = 0
    while i + n <= len(lst):
        yield lst[i:i + n]
        i += scrolling

def label_for_windowing(labels, n=TIME_STEPS, scrolling = SCROLLING):
    if(n == scrolling):
        return labels[(n-1):]

    labels_ret = []
    first = False

    for i in range(len(labels)):
        if(first):
            if(i%scrolling == 0):
                labels_ret.append(labels[i])

        if i == n-1:
            labels_ret.append(labels[i])
            first = True
    
    return labels_ret
        
def aggregate_data(folder):
    X = []
    y = []
    i=0

    for root, dirs, files in os.walk(folder):
        for name_file in files:
            print(i)

            npy_path = os.path.join(root, name_file)
            data = np.load(npy_path).tolist()

            data_X, data_y = split_X_y(data)

            rest = len(data_X) % TIME_STEPS
            if (rest != 0):
                data_X = data_X[:-rest]
                data_y = data_y[:-rest]

            for el in data_X:
                X.append(el)

            for el in data_y:
                y.append(el)
            i+=1

    return np.array(X), np.array(y)

def load_data(): 
    npy_folder_train = 'path numpy array training set'
    npy_folder_eval = 'path numpy array evaluation set'
    npy_folder_test = 'path numpy array testing set'

    print("Sliding windows training set")
    X_train, y_train = aggregate_data(npy_folder_train)
    print("Sliding windows evaluation set")
    X_eval, y_eval = aggregate_data(npy_folder_eval)
    print("Sliding windows testing set")
    X_test, y_test = aggregate_data(npy_folder_test)

    X_train, X_eval, X_test, y_train, y_eval, y_test = sequentioning_overlapping(X_train, X_eval, X_test, y_train, y_eval, y_test)

    return X_train, X_eval, X_test, y_train, y_eval, y_test

def labeling_sequence(labels):
    labels = labels.tolist()
    y = []

    for i in range(len(labels)):
        if ((i+1)%TIME_STEPS == 0):
            y.append(labels[i])

    return np.array(y)

def gen_to_list(gen):
    res = []
    for el in gen:
        res.append(el)
    return res

def sequentioning_overlapping(X_train, X_eval, X_test, y_train, y_eval, y_test):
    X_train = gen_to_list(sliding_windows(X_train))
    X_eval = gen_to_list(sliding_windows(X_eval))
    X_test = gen_to_list(sliding_windows(X_test))

    y_train = label_for_windowing(y_train)
    y_eval = label_for_windowing(y_eval)
    y_test = label_for_windowing(y_test)

    return X_train, X_eval, X_test, y_train, y_eval, y_test

def create_model(hyperparam):
    lstm_units = hyperparam.Int('units', min_value=6, max_value=120, step=6)
    lstm_activation = hyperparam.Choice('lstm_activation', values=['relu', 'tanh', 'sigmoid'], default='tanh')
    drop_rate = rate= hyperparam.Float('dropout_1', min_value=0.0, max_value=0.5, default=0.25,step=0.05)
    dense_activation = hyperparam.Choice('dense_activation', values=['relu', 'softmax', 'sigmoid'], default='sigmoid')
    learning_rate = hyperparam.Float('learning_rate',min_value=1e-4,max_value=1e-2,sampling='LOG',default=1e-3)

    model = Sequential()
    model.add(LSTM(lstm_units, activation=lstm_activation, input_shape=(TIME_STEPS,NUM_FEATURES)))
    model.add(Dropout(rate=drop_rate))
    model.add(Dense(1, activation=dense_activation))
    model.compile(optimizer=optimizers.Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    X_train, X_eval, X_test, y_train, y_eval, y_test = load_data()

    X_train = np.array(X_train)
    X_eval = np.array(X_eval)
    X_test = np.array(X_test)

    y_train = np.array(y_train)
    y_eval = np.array(y_eval)
    y_test = np.array(y_test)

    tuner = Hyperband(create_model,objective='val_loss',max_epochs=100,factor=3,directory='test_val_loss',project_name='drowsiness_detection')
    stop_early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    
    print("\n\n*****************SEARCH SPACE SUMMARY*****************\n\n")
    print(tuner.search_space_summary())

    tuner.search(X_train, y_train, epochs=100, validation_data=(X_eval, y_eval), callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    print(tuner.results_summary())


if __name__ == "__main__":
    main()