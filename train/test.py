import pickle
from matplotlib import pyplot as plt

from utils import forward_prop, get_predictions, data_set

X_train, Y_train, _ = data_set()

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

with open("model/mnist.pkl", 'rb') as f:
  model = pickle.load(f)
  W1 = model['W1']
  b1 = model['b1']
  W2 = model['W2']
  b2 = model['b2']

  test_prediction(0, W1, b1, W2, b2)
  test_prediction(1, W1, b1, W2, b2)
  test_prediction(2, W1, b1, W2, b2)
  test_prediction(3, W1, b1, W2, b2)
  test_prediction(4, W1, b1, W2, b2)
  test_prediction(5, W1, b1, W2, b2)
  test_prediction(0, W1, b1, W2, b2)