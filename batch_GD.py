import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(o):
    temp = np.exp(o)
    (r,c) = np.shape(o)
    for i in range(c):
        sum = np.sum(temp[:,i])
        temp[:,i] = temp[:,i]/sum
    
    return temp

def sigmoid_prime(x):
    return np.exp(-x)/((1+np.exp(-x))**2)

no_epoch = 500
learn_rate = 0.01
# Imgae dimension
D = 50
# Number of hidden units
H = 200
# Number of classes
C = 10

# Initialize weights
np.random.seed(1)
# W should be uniform between [-u,u]
#u = np.sqrt(6/H)
#u=0.01
#W1 = np.random.uniform(-u,u,(H,D**2))
#W2 = np.random.uniform(-u,u, (C,H))
W1 = np.random.normal(0,0.15,(H,D**2))
W2 = np.random.normal(0,0.15, (C,H))
# b should be initialised to 0
b1 = np.zeros((H,1))
b2 = np.zeros((C,1))

# Input Matrix, 1568 x 20000
N_total = 20000
N_train = 10000
N_val = 5000
N_test = 5000

x_train = flat_Xn[0:N_train,:]
x_train= x_train.T
x_val = flat_Xn[N_train:N_train+N_val,:]
x_val = x_val.T
x_test = flat_Xn[N_train+N_val:N_total,:]
x_test = x_test.T

# One-hot encoding
y_hot = np.zeros((C,N_total))
for i in range(N_total):
    y_hot[y_label[0,i],i] = 1

y = y_hot[:,0:N_train]
y_val = y_hot[:,N_train:N_train+N_val]
y_test = y_hot[:,N_train+N_val:N_total]

avg_loss_train, avg_loss_val, avg_loss_test = [], [], []
avg_class_err_train, avg_class_err_val, avg_class_err_test = [], [], []
for epoch in range(no_epoch):
    #== TRAINING ==================================================================
    q = np.matmul(W1,x_train) + b1
    h = sigmoid(q)
    o = np.matmul(W2,h) + b2
    p = softmax(o)
         
    dL_dW2 = (1./N_train)*np.multiply(1,np.matmul((p-y),h.T))
    W2_new = W2 - learn_rate*dL_dW2
    
    dL_dW1 = (1./N_train)*np.multiply(1,np.matmul(np.matmul(W2.T, p-y)*sigmoid_prime(q),x_train.T))
    W1_new = W1 - learn_rate*dL_dW1
    
    dL_db2 = (1./N_train)*np.sum(p-y, axis=1, keepdims=True)
    b2_new = b2 - learn_rate*dL_db2
        
    dL_db1 = (1./N_train)*np.sum(np.matmul(W2.T,p-y)*sigmoid_prime(q), axis=1, keepdims=True)
    b1_new = b1 - learn_rate*dL_db1
     
    # Cross-entropy Loss
    loss_train = 0    
    for i in range(N_train):
        loss_train = loss_train + np.dot(y[:,i].T, np.log(1/p[:,i]))
    avg_loss_train += [loss_train/N_train];
    
    # Classification error
    no_err_train = 0
    for i in range(N_train):
        max_p_pos = np.argmax(p[:,i])
        if y[max_p_pos,i] != 1:
            no_err_train += 1
    avg_class_err_train += [100*no_err_train/N_train]
    
    #== VALIDATION ==============================================================
    q_val = np.matmul(W1,x_val) + b1
    h_val = sigmoid(q_val)
    o_val = np.matmul(W2,h_val) + b2
    p_val = softmax(o_val)  # - Predicted label
       
    # Cross-entropy loss
    loss_val = 0    
    for i in range(N_val):
        loss_val = loss_val + np.dot(y_val[:,i].T, np.log(1/p_val[:,i]))
    avg_loss_val += [loss_val/N_val];
    
    # Classification error
    no_err_val = 0
    for i in range(N_val):
        max_p_pos = np.argmax(p_val[:,i])
        if y_val[max_p_pos,i] != 1:
            no_err_val += 1
    avg_class_err_val += [100*no_err_val/N_val]
    
    #== Testing ==============================================================
    q_test = np.matmul(W1,x_test) + b1
    h_test = sigmoid(q_test)
    o_test = np.matmul(W2,h_test) + b2
    p_test = softmax(o_test)  # - Predicted label
       
    # Cross-entropy loss
    loss_test = 0    
    for i in range(N_test):
        loss_test = loss_test + np.dot(y_test[:,i].T, np.log(1/p_test[:,i]))
    avg_loss_test += [loss_test/N_test];
    # Classification error
    no_err_test = 0
    for i in range(N_test):
        max_p_pos = np.argmax(p_test[:,i])
        if y_test[max_p_pos,i] != 1:
            no_err_test += 1
    avg_class_err_test += [100*no_err_test/N_test]
    #==============================================================================
    
    #    Updtae the weights and biases to the new values
    W1 = W1_new
    W2 = W2_new
    b1 = b1_new
    b2 = b2_new
    
    
    print('\nEpoch = ', epoch)
    print('Train Loss = ', avg_loss_train[epoch])
    print('Val Loss = ', avg_loss_val[epoch])
    print('Test Loss = ', avg_loss_test[epoch])
    
    print('Train error = ', avg_class_err_train[epoch])
    print('Val error = ', avg_class_err_val[epoch])
    print('Test error = ', avg_class_err_test[epoch])

# Plot Cross Entropy Loss
plt.figure()  
plt.plot(range(0,no_epoch), avg_loss_train, label = 'Training Loss')
plt.plot(range(0,no_epoch), avg_loss_val, c = 'r', label = 'Validation Loss')
plt.plot(range(0,no_epoch), avg_loss_test, c = 'g', label = 'Testing Loss')
plt.legend(loc = 'upper right')
plt.xlabel('Epoch')
plt.ylabel('Cross-entropy Loss')

# Plot Classification Error 
plt.figure()   
plt.plot(range(0,no_epoch), avg_class_err_train, label = 'Training error')
plt.plot(range(0,no_epoch), avg_class_err_val, c = 'r', label = 'Validation error')
plt.plot(range(0,no_epoch), avg_class_err_test, c = 'g', label = 'Testing error')
plt.legend(loc = 'upper right')
plt.xlabel('Epoch')
plt.ylabel('Percentage Classification Error')
       
    
    
        
