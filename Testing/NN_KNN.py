import numpy as np
import cv2
import tensorflow as tf
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


dict_classes = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}


def most_frequent(list):
    list = [x[0] for x in list]
    return [max(set(list), key = list.count)]


def predictLabelNN(x_train_flatten, y_train, img):
    predictedLabel = -1
    scoreMin = 100000000
    score = 0

    for idx, imgT in enumerate(x_train_flatten):
        difference = np.abs(img - imgT)
        score = np.sum(difference)
        if score < scoreMin:
            scoreMin = score
            predictedLabel = y_train[idx]

    return predictedLabel


def predictLabelKNN(x_train_flatten, y_train, img):

    predictedLabel = -1
    predictions = []  # list to save the scores and associated labels as pairs  (score, label)
    score = 0

    for idx, imgT in enumerate(x_train_flatten):

        difference = np.abs(img - imgT)
        score = np.sum(difference)
        predictions.append((score, y_train[idx]))

    predictions.sort(key=lambda x: x[0])
    k = 10
    predictions = predictions[:k]

    predLabels = [predictions[i][1] for i in range(len(predictions))]


    #  - Application 2 - Step 1h - Determine the dominant class from the predicted labels
    predictedLabel = most_frequent(predLabels)

    return predictedLabel
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
def main():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    #  - Exercise 1 - Determine the size of the four vectors x_train, y_train, x_test, y_test
    #print("The size of x_train", len(x_train))
    #print("The size of y_train", len(y_train))
    #print("The size of x_test", len(x_test))
    #print("The size of y_test", len(y_test))
    #print(len(x_train[0][0][0]))
    #  - Exercise 2 - Visualize the first 10 images from the testing dataset with the associated labels
    x_train_flatten = np.float64(x_train.reshape(x_train.shape[0], 32 * 32 * 3))
    x_test_flatten = np.float64(x_test.reshape(x_test.shape[0], 32 * 32 * 3))
    numberOfCorrectPredictedImages = 0
    for idx, img in enumerate(x_test_flatten[0:200]):
        predictedLabel = predictLabelKNN(x_train_flatten, y_train, img)

        if y_test[idx] == predictedLabel:
            numberOfCorrectPredictedImages += 1

    accuracy = 100*(numberOfCorrectPredictedImages/200)

    print("System accuracy = {}".format(accuracy))


    return
#####################################################################################################################
#####################################################################################################################




#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################




#####################################################################################################################
#####################################################################################################################

