from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

N_total = 20000
N_train = 10000
N_val = 5000
N_test = 5000

x_train = flat_Xn[0:N_train,:]
x_val = flat_Xn[N_train:N_train+N_val,:]
x_test = flat_Xn[N_train+N_val:N_total,:]
#x_test = x_test.T

y_train = Pc[:,0:N_train].T
y_val = Pc[:,N_train:N_train+N_val].T
y_test = Pc[:,N_train+N_val:N_total].T

y_test_class = y_label[:,N_train+N_val:N_total].T


model = keras.Sequential([
    keras.layers.Dense(200, activation=tf.nn.relu),
    keras.layers.Dense(175, activation=tf.nn.relu),
    keras.layers.Dense(150, activation=tf.nn.relu),
    keras.layers.Dense(125, activation=tf.nn.relu),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(75, activation=tf.nn.relu),
    keras.layers.Dense(50, activation=tf.nn.relu),
    keras.layers.Dense(25, activation=tf.nn.relu),
    keras.layers.Dense(15, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer = 'adam', 
              loss = 'mean_squared_error',
              metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 100)
test_loss, test_acc = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)

y_pred_class = np.zeros(y_pred.shape)
for i in range(len(y_pred)):
    if y_pred[i,:] == 1:
        y_pred_class[i,...] = 0
    else:
        y_pred_class[i,...] = 9 - np.floor(y_pred[i,:]/0.1)
        
accuracy = np.sum(y_pred_class==y_test_class)/N_test
print('Classification Accuracy:', accuracy)
#fig, ax = plt.subplots()
#ax.scatter(y_test, y_pred)
#ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
#ax.set_xlabel('Measured')
#ax.set_ylabel('Predicted')
#plt.show()

#accuracy, loss = [],[]
#for i in range(100):
#    model.fit(x_train, y_train, epochs=i)
#    test_loss, test_acc = model.evaluate(x_test, y_test)
#    accuracy += [test_acc]
#    loss += [test_loss]
#    print('Epoch:', i)
#    print('Test accuracy:', test_acc)

# Plot Classification Accuracy
#plt.figure()   
#plt.plot(range(0,100), accuracy, label = 'Testing accuracy')
#plt.xlabel('Epoch')
#plt.ylabel('Testing accuracy')





