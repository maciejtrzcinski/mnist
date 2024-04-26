import numpy as np
import pandas as pd

def data_set(path = 'train/digit-recognizer/train.csv'):
    data = pd.read_csv(path)

    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)

    data_dev = data[0:1000].T
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.
    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.

    return X_train, Y_train, m