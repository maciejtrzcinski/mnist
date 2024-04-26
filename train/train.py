import pickle
import time
import numpy as np

from utils import backward_prop, forward_prop, get_predictions, init_params, data_set

X_train, Y_train, m = data_set()

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    tic = time.perf_counter()
    W1, b1, W2, b2 = init_params()
    d = dict()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    toc = time.perf_counter()
    print(f"Finished training in {toc - tic:0.4f} seconds")
    d["W1"] = W1
    d["b1"] = b1
    d["W2"] = W2
    d["b2"] = b2
    with open("model/mnist.pkl", 'wb') as f:
        pickle.dump(d, f)

gradient_descent(X_train, Y_train, 0.10, 1500)
