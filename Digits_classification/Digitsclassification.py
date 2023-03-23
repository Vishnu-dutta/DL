import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(len(X_train))
print(len(X_test))
'''
checking each individual sample we get 28x28 pixel image and they represent a 2D array
'''
print(X_train[0].shape)  # shape of the element at position index 0
print(X_train[0])

'''
plotting the first training image X_train[0] and y_train gives us 5
'''
plt.imshow(X_train[59999])
print(y_train[59999])
plt.show()

X_train = X_train / 255  # dividing by 255 to limit the range from 0 to 1
X_test = X_test / 255

print(X_train[0])  # to get the 2D array

print(X_train.shape)
'''
conversion of the 2D array to 1D array because in presentation we want to convert 28x28 image to this single dimensional 
array that will have 784 elements. In pandas we have this function called 'reshape'. At first in X_train.shape we have 
(60000, 28, 28) even after reshape we still want (60000). Storing it in the variable X_train_flattened and 
X_test_flattened
'''
X_train_flattened = X_train.reshape(len(X_train), 28 * 28)
X_test_flattened = X_test.reshape(len(X_test), 28 * 28)

print(X_train_flattened.shape)
print(X_test_flattened.shape)

print(X_train_flattened[0])

'''
Creating the simple neural network. With 784 as the input layer and 10 as the output layer. 
And 'Dense' means all the neurons in one layer are connected with every other neuron in the second layer. 
'''

model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)

model.evaluate(X_test_flattened, y_test)

'''
X_test_flattened we can get direct images and samples from this dataset. But to get the desired results from the 
predicting, y_predicted we get set of 10 results from there the highest value is the result. 
np.argmax(y_predicted[0]) == plt.imshow(X_test[0]) at least for the prediction part
'''

y_predicted = model.predict(X_test_flattened)

print(y_predicted[0])

plt.imshow(X_test[0])
np.argmax(y_predicted[0])  # we get 7

y_predicted_labels = [np.argmax(i) for i in y_predicted]
print(y_predicted_labels[:5])  # we get [7, 2, 1, 0, 4]

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
print(cm)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

'''
adding a hidden layer with 100
'''

model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)

print(model.evaluate(X_test_flattened, y_test))

y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)

print(model.evaluate(X_test, y_test))
