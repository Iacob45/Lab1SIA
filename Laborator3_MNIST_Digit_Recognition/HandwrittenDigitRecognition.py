import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
import numpy as np
#####################################################################################################################
#####################################################################################################################

#output = ((input - kernel + 2 * padding) // stride)+1

#####################################################################################################################
#####################################################################################################################
def baseline_model(num_pixels, num_classes):

    #TODO - Application 1 - Step 6a - Initialize the sequential model
    model = keras.models.Sequential()

    #TODO - Application 1 - Step 6b - Define a hidden dense layer with 8 neurons
    model.add(layers.Dense(units=16, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))

    #TODO - Application 1 - Step 6c - Define the output dense layer
    model.add(layers.Dense(units=num_classes, kernel_initializer='normal', activation='softmax'))

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
    model = keras.models.Sequential()

    #TODO - Application 2 - Step 5b - Create the first hidden layer as a convolutional layer
    model.add(layers.Conv2D(filters=9, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

    #TODO - Application 2 - Step 5c - Define the pooling layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    #TODO - Application 2 - Step 5d - Define the Dropout layer
    model.add(layers.Dropout(rate=0.2))  # 20% dropout

    #TODO - Application 2 - Step 5e - Define the flatten layer
    model.add(layers.Flatten())

    #TODO - Application 2 - Step 5f - Define a dense layer of size 128
    model.add(layers.Dense(units=128, kernel_initializer='normal', activation='relu'))

    #TODO - Application 2 - Step 5g - Define the output layer
    model.add(layers.Dense(units=num_classes, kernel_initializer='normal', activation='softmax'))

    #TODO - Application 2 - Step 5h - Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def CNN_model2(input_shape, num_classes):

    # TODO - Application 2 - Step 5a - Initialize the sequential model
    model = keras.models.Sequential()

    #TODO - Application 2 - Step 5b - Create the first hidden layer as a convolutional layer
    model.add(layers.Conv2D(filters=30, kernel_size=(5, 5), activation='relu', input_shape=input_shape))

    #TODO - Application 2 - Step 5c - Define the pooling layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(filters=15, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    #TODO - Application 2 - Step 5d - Define the Dropout layer
    model.add(layers.Dropout(rate=0.2))  # 20% dropout

    #TODO - Application 2 - Step 5e - Define the flatten layer
    model.add(layers.Flatten())

    #TODO - Application 2 - Step 5f - Define a dense layer of size 128
    model.add(layers.Dense(units=128, kernel_initializer='normal', activation='relu'))
    #model.add(layers.Dense(units=50, kernel_initializer='normal', activation='relu'))

    #TODO - Application 2 - Step 5g - Define the output layer
    model.add(layers.Dense(units=num_classes, kernel_initializer='normal', activation='softmax'))

    #TODO - Application 2 - Step 5h - Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def trainAndPredictCNN(X_train, Y_train, X_test, Y_test):

    #TODO - Application 2 - Step 2 - reshape the data to be of size [samples][width][height][channels]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[1], 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[1], 1).astype('float32')

    #TODO - Application 2 - Step 3 - normalize the input values from 0-255 to 0-1
    X_train /= 255
    X_test /= 255

    #TODO - Application 2 - Step 4 - One hot encoding - Transform the classes labels into a binary matrix
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    num_classes = Y_train.shape[1]

    #TODO - Application 2 - Step 5 - Call the cnn_model function
    model = CNN_model2(X_train.shape[1:],num_classes)


    #TODO - Application 2 - Step 6 - Train the model
    model.fit(x=X_train, y=Y_train, batch_size=200, epochs=5, verbose=2, validation_data=(X_test, Y_test), shuffle=True)

    #TODO - Application 2 - Step 8 - Final evaluation of the model - compute and display the prediction error
    scores = model.evaluate(x=X_test, y=Y_test, verbose=0)
    print("CNN Error: {:.2f}".format(100 - scores[1] * 100))


    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():

    #TODO - Application 1 - Step 1 - Load the MNIST dataset in Keras
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()


    #TODO - Application 1 - Step 2 - Train and predict on a MLP - Call the trainAndPredictMLP function
    #trainAndPredictMLP(X_train, Y_train, X_test, Y_test)

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
