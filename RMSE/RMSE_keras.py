from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)



def split_100(x, label):
    x_new = np.zeros((100,28,28))
    label_new = np.zeros(100)
    count = np.zeros(10)
    
    points = 0
    while points != 100:
        
        rand_pt = np.random.randint(0,x.shape[0])
        
        if(count[label[rand_pt]] < 10):
            
            count[label[rand_pt]] += 1
            x_new[points,:,:] = x[rand_pt]
            label_new[points] = label[rand_pt]
            
            x = np.delete(x,rand_pt,0)
            label = np.delete(label,rand_pt,0)
            
            points += 1
    
    labels = np.uint8(label_new)
    return x_new, labels


train_images_100, train_labels_100 = split_100(train_images, train_labels)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

accuracy, loss = [],[]
for i in range(100):
    model.fit(train_images_100, train_labels_100, epochs=i)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    accuracy += [test_acc]
    loss += [test_loss]
    print(i)
    
    
print('Best Test accuracy:', accuracy[np.argmax(accuracy)])
print('At Epoch:',np.argmax(accuracy))

# Plot Classification Accuracy
plt.figure()   
plt.plot(range(0,100), accuracy, label = 'Testing accuracy')
plt.xlabel('Epoch')
plt.ylabel('Testing accuracy')





