from __future__ import absolute_import, division, print_function

#importing libraries used in neural networks
import tensorflow as tf
from tensorflow import keras

#importing Python's libraries
import matplotlib.pyplot as plt
import numpy as np

#defining the datasets
fashion_mnist = keras.datasets.fashion_mnist

#arrays derived from the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#creating names for labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#The code below is used to look into the first image of the
#dataset to test if the program returns the image with pixel values
# plt.figure()
# plt.imshow(train_images) ##you can look up a particular image of the dataset by searching for its index using plt.imshow(train_images[5]), which in this case the code will return the sixth image within the dataset, which is a pullover.   
# plt.colorbar()
# plt.grid(False)
# plt.show()

#To train neural network the dataset's images need to be sclaed into a value between 0 or 1
#so the values are divided by 255
train_images = train_images / 255.0
test_images = test_images / 255.0

#testing if labels of the images are correct
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#configuring the layers of the neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#code used to measure how accurate is the model during training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#at this step the network starts its training
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

#prining out test to indicate the accuracy of a test
print('Test accuracy: ', test_acc)

predictions = model.predict(test_images)

print(predictions[2])#indicates the number of image to test

#prining out the text to indicate the gategory  of a prediction
print("The predicted category is: " + str(np.argmax(predictions[2])) + ", which is: " + class_names[
    int(np.argmax(predictions[2]))])

#prining out the text to indicate the category of the correct category
print("The correct category is: " + str(test_labels[2]) + ", which is: " + class_names[int(np.argmax(predictions[2]))])

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array), class_names[true_label]), color=color)

#creating graph to display information from the tests
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(range(10), class_names, rotation=90)
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = "#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()


#the code used to indidate the number of an image to check
def select_image(predictions, test_labels, test_images):
    i = input("Which index would you like to check? (0 - 10000)")
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(int(i), predictions, test_labels, test_images)
    plt.subplot(1,2,2)
    plot_value_array(int(i), predictions, test_labels)
    plt.show()
    select_image(predictions, test_labels, test_images)

#singleline display of the checked graphs
#select_image(predictions, test_labels, test_images)

def multi_plot(predictions, test_labels, test_image):
    num_rows = int(input("How many rows would you like?"))
    num_cols = int(input("How many columns would you like?"))
    num_images = num_rows * num_cols
    plt.figure(figsize=(2*2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels)
    plt.show()

#multiline display of the checked graphs
multi_plot(predictions, test_labels, test_images)

