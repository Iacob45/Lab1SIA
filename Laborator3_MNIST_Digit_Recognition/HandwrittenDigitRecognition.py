import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
import numpy as np
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def baseline_model(num_pixels, num_classes):

    #TODO - Application 1 - Step 6a - Initialize the sequential model
    model = keras.models.Sequential()

    #TODO - Application 1 - Step 6b - Define a hidden dense layer with 8 neurons
    model.add(layers.Dense(8, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))

    #TODO - Application 1 - Step 6c - Define the output dense layer
    model.add(layers.Dense(num_classes, kernel_initializer='normal', activation='softmax'))

    # TODO - Application 1 - Step 6d - Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def trainAndPredictMLP(X_train, Y_train, X_test, Y_test):

    #TODO - Application 1 - Step 3 - Reshape the MNIST dataset - Transform the images to 1D vectors of floats (28x28 pixels  to  784 elements)
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

    #TODO - Application 1 - Step 4 - Normalize the input values
    X_train = X_train / 255
    X_test = X_test / 255


    #TODO - Application 1 - Step 5 - Transform the classes labels into a binary matrix
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    num_classes = Y_test.shape[1]



    #TODO - Application 1 - Step 6 - Build the model architecture - Call the baseline_model function
    model = baseline_model(num_pixels, num_classes)


    #TODO - Application 1 - Step 7 - Train the model
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)

    #TODO - Application 1 - Step 8 - System evaluation - compute and display the prediction error
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Baseline Error: {:.2f}".format(100 - scores[1] * 100))


    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def CNN_model(input_shape, num_classes):

    # TODO - Application 2 - Step 5a - Initialize the sequential model
    model = None   #Modify this


    #TODO - Application 2 - Step 5b - Create the first hidden layer as a convolutional layer


    #TODO - Application 2 - Step 5c - Define the pooling layer


    #TODO - Application 2 - Step 5d - Define the Dropout layer


    #TODO - Application 2 - Step 5e - Define the flatten layer


    #TODO - Application 2 - Step 5f - Define a dense layer of size 128


    #TODO - Application 2 - Step 5g - Define the output layer


    #TODO - Application 2 - Step 5h - Compile the model


    return model
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def trainAndPredictCNN(X_train, Y_train, X_test, Y_test):

    #TODO - Application 2 - Step 2 - reshape the data to be of size [samples][width][height][channels]


    #TODO - Application 2 - Step 3 - normalize the input values from 0-255 to 0-1


    #TODO - Application 2 - Step 4 - One hot encoding - Transform the classes labels into a binary matrix


    #TODO - Application 2 - Step 5 - Call the cnn_model function
    model = None   #Modify this


    #TODO - Application 2 - Step 6 - Train the model


    #TODO - Application 2 - Step 8 - Final evaluation of the model - compute and display the prediction error



    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():

    #TODO - Application 1 - Step 1 - Load the MNIST dataset in Keras
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()


    #TODO - Application 1 - Step 2 - Train and predict on a MLP - Call the trainAndPredictMLP function
    trainAndPredictMLP(X_train, Y_train, X_test, Y_test)


    #TODO - Application 2 - Step 1 - Train and predict on a CNN - Call the trainAndPredictCNN function
    trainAndPredictCNN(X_train, Y_train, X_test, Y_test)


    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
