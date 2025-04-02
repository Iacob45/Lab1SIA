#####################################################################################################################
#####################################################################################################################
import numpy as np

#####################################################################################################################
#####################################################################################################################
def activationFunction(n):

    #TODO - Application 1 - Step 4b - Define the binary step function as activation function
    if n >= 0:
        return 1
    else:
        return 0

#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def forwardPropagation(x, weights, bias):

    a = None # the neuron output
    # TODO - Application 1 - Step 4a - Multiply weights with the input vector (p) and add the bias   =>  n
    out = x[0]*weights[0] + x[1]*weights[1] + bias

    # TODO - Application 1 - Step 4c - Pass the result to the activation function  =>  a
    a = activationFunction(out)


    return a
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():

    #Application 1 - Train a single neuron perceptron in order to predict the output of an AND gate.
    #The network should receive as input two values (0 or 1) and should predict the target output


    #Input data
    X = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]

    #Labels
    t = [0, 0, 0, 1]

    #TODO - Application 1 - Step 2 - Initialize the weights with zero  (weights)
    w = np.zeros(2)

    #TODO - Application 1 - Step 2 - Initialize the bias with zero  (bias)
    b = 0

    #TODO - Application 1 - Step 3 - Set the number of training steps  (epochs)
    epochs = 5

    #TODO - Application 1 - Step 4 - Perform the neuron training for multiple epochs
    for ep in range(epochs):
        for i in range(len(t)):
            x = X[i]
            #TODO - Application 1 - Step 4 - Call the forwardPropagation method
            out = forwardPropagation(x, w, b)
            #TODO - Application 1 - Step 5 - Compute the prediction error (error)
            error = t[i]-out

            #TODO - Application 1 - Step 6 - Update the weights
            w[0] = w[0] + error * x[0]
            w[1] = w[1] + error * x[1]

            #TODO - Application 1 - Step 7 - Update the bias
            b = b + error

            # DELETE THIS
            continue

    #TODO - Application 1 - Step 8 - Print weights and bias
    print("Weights: ",w)
    print("Bias: ", b)


    # TODO - Application 1 - Step 9 - Display the results
    for i in X:
        out = forwardPropagation(i,w,b)
        print("Intrare: ", i, "Predictie: ",out)

   
    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == "__main__":
    main()
#####################################################################################################################
#####################################################################################################################