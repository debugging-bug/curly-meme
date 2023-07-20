import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')

data = np.array(data)  # convert the data into an array
m, n = data.shape
np.random.shuffle(data)  # shuffle before splitting into dev and training sets

#here we mess with the data a bit
data_dev = data[0:1000].T # transpose the data
Y_dev = data_dev[0] # the first row is the labels
X_dev = data_dev[1:n] # the rest of the rows are the data
X_dev = X_dev / 255. # normalize our data

data_train = data[1000:m].T  # transpose the data
Y_train = data_train[0] 
X_train = data_train[1:n] 
X_train = X_train / 255. 
_, m_train = X_train.shape # m_train is the number of examples in the training set

#we initiate the bias weights and bla bla they will be tweaked later


def init_params(): # initialize the weights and biases randomly
    W1 = np.random.rand(10, 784) - 0.5 # multiply by 2 and subtract by 0.5 to get values between -1 and 1
    b1 = np.random.rand(10, 1) - 0.5 # we have 10 nodes in the hidden layer and 784 input nodes
    W2 = np.random.rand(10, 10) - 0.5 # we have 10 nodes in the output layer and 10 hidden nodes
    b2 = np.random.rand(10, 1) - 0.5 
    # print(W1, b1, W2, b2)
    return W1, b1, W2, b2 # return the initialized weights and biases as a tuple


#the relu activation function 
def ReLU(Z): 
    return np.maximum(Z, 0)  # return the element-wise maximum of Z and 0

#the softmax avtivation function


def softmax(Z):   
    A = np.exp(Z) / sum(np.exp(Z)) 
    return A

#here is the forward propogation


def forward_prop(W1, b1, W2, b2, X): 
    Z1 = W1.dot(X) + b1  # the first layer affine function
    A1 = ReLU(Z1)  # activation function of the first layer
    Z2 = W2.dot(A1) + b2  # second layer affine function
    A2 = softmax(Z2)  # softmax function of the second layer
    return Z1, A1, Z2, A2


def ReLU_deriv(Z):
    return Z > 0 # returns a boolean array

#defining y


def one_hot(Y): 
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))  # Y.max() + 1 is the number of labels
    one_hot_Y[np.arange(Y.size), Y] = 1 # for each row, change the column with index Y to 1
    one_hot_Y = one_hot_Y.T # transpose the array
    return one_hot_Y    

#backward propogation to tweak the bias and the wieghts


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y): 
    one_hot_Y = one_hot(Y) # convert the label array to a one hot matrix
    dZ2 = A2 - one_hot_Y # the error for the second layer
    dW2 = 1 / m * dZ2.dot(A1.T) # the error derivative for W2
    db2 = 1 / m * np.sum(dZ2) # the error derivative for b2
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1) # the error for the first layer
    dW1 = 1 / m * dZ1.dot(X.T) # the error derivative for W1
    db1 = 1 / m * np.sum(dZ1) # the error derivative for b1
    return dW1, db1, dW2, db2

#now we update the weights and the bias


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1 # alpha is the learning rate
    b1 = b1 - alpha * db1 # update the weights and biases
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2
    # print(W1, b1, W2, b2)
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0) # return the indices of the max values in each column


def get_accuracy(predictions, Y):
    print(predictions, Y) # print the predictions and the actual labels
    return np.sum(predictions == Y) / Y.size # return accuracy on a single batch

#gradient_descent


def gradient_descent(X, Y, alpha, iterations): 
    W1, b1, W2, b2 = init_params() # initialize the weights and biases
    for i in range(iterations): # iterate
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X) # forward prop
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y) # backward prop
        W1, b1, W2, b2 = update_params(
            W1, b1, W2, b2, dW1, db1, dW2, db2, alpha) # update parameters
        if i % 10 == 0: # print the accuracy every 10 iterations
            print("Iteration: ", i) 
            predictions = get_predictions(A2) # get the predictions
            print(get_accuracy(predictions, Y)) # get the accuracy
    return W1, b1, W2, b2 


#predict the accuray trainign the model
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 200) # train the model
print("Completed") 


#make predictions from images
def make_predictions(X, W1, b1, W2, b2): 
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X) # forward prop
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2): 
    current_image = X_train[:, index, None] # get the current image
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2) # make prediction
    label = Y_train[index] # get the label
    print("Prediction: ", prediction) # print the prediction
    print("Label: ", label)
    print(get_accuracy(prediction, label)) # print the accuracy

    current_image = current_image.reshape((28, 28)) * 255 # reshape the image from (784, 1) to (28, 28)
    plt.gray() # set colormap to gray
    plt.imshow(current_image, interpolation='nearest') # show the image
    plt.show()
 

test_prediction(6969, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2) 
test_prediction(200, W1, b1, W2, b2)
test_prediction(69, W1, b1, W2, b2)
test_prediction(444, W1, b1, W2, b2)
test_prediction(54, W1, b1, W2, b2)
test_prediction(129, W1,  b1, W2, b2)
test_prediction(43, W1, b1, W2, b2)
test_prediction(93, W1, b1, W2, b2)
test_prediction(33, W1, b1, W2, b2)
test_prediction(22, W1, b1, W2, b2)
test_prediction(12, W1, b1, W2, b2)
 