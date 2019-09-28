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

# Number of Epochs
no_epoch = 50
# learning rate
learn_rate = 0.01
# Momentum
beta = 0.5
# Imgae dimension
D = 50
# Number of hidden units
H1, H2 = 100, 100
# Number of classes
C = 10

u = np.sqrt(6/(H1+H2))
np.random.seed(1)
W1 = np.random.uniform(-u,u,(H1,D**2))
W2 = np.random.uniform(-u,u, (H2,H1))
W3 = np.random.uniform(-u,u, (C,H2))
W1 = np.random.normal(0,0.15,(H1,D**2))
W2 = np.random.normal(0,0.15, (H2,H1))
W3 = np.random.normal(0,0.15, (C,H2))
# b should be initialised to 0
b1 = np.zeros((H1,1))
b2 = np.zeros((H2,1))
b3 = np.zeros((C,1))

# Input Matrix, 1568 x 20000
N_total = 20000
N_train = 10240
N_val = 5000
N_test = 4760

# no of batch = 20000/32 = 625
batch_size = 32
no_batches = N_train//batch_size

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

# Predicted output (or the probability) 19x2000
p_final = np.zeros((C,N_train))

prev_dL_dW1, prev_dL_dW2, prev_dL_dW3 = 0,0,0
prev_dL_db1, prev_dL_db2, prev_dL_db3 = 0,0,0
avg_err_train, avg_err_val, avg_err_test = [], [], []
avg_class_err_train, avg_class_err_val, avg_class_err_test = [], [], []

for epoch in range(no_epoch):
    
# Mini batch gradient descent
    for b_no in range(no_batches):
        
        input_x_batch = x_train[:, b_no*32: (b_no*32 + 32)]
        y_batch = y[:, b_no*32: (b_no*32 + 32)]
        
    #== TRAINING =============================================================
        q1 = np.matmul(W1,input_x_batch) + b1
        h1 = sigmoid(q1)
        q2 = np.matmul(W2,h1) + b2
        h2 = sigmoid(q2)
        o = np.matmul(W3,h2) + b3
        p = softmax(o)  
    #=========================================================================
        dL_dW3 = (1./batch_size)*np.multiply(1,np.matmul((p-y_batch),h2.T))
        momn_dL_dW3 = dL_dW3 + beta*prev_dL_dW3
        W3_new = W3 - learn_rate*momn_dL_dW3
        prev_dL_dW3 = momn_dL_dW3
        
        dL_dW2 = (1./batch_size)*np.multiply(1,np.matmul(np.matmul(W3.T, p-y_batch)*sigmoid_prime(q2),h1.T))
        momn_dL_dW2 = dL_dW2 + beta*prev_dL_dW2
        W2_new = W2 - learn_rate*momn_dL_dW2
        prev_dL_dW2 = momn_dL_dW2
        
        dL_dW1 = (1./batch_size)*np.multiply(1,np.matmul(np.matmul(W2.T,np.matmul(W3.T, p-y_batch))*sigmoid_prime(q2)*sigmoid_prime(q1),input_x_batch.T))
        momn_dL_dW1 = dL_dW1 + beta*prev_dL_dW1
        W1_new = W1 - learn_rate*momn_dL_dW1
        prev_dL_dW1 = momn_dL_dW1
        
        dL_db3 = (1./batch_size)*np.sum(p-y_batch, axis=1, keepdims=True)
        momn_dL_db3 = dL_db3 + beta*prev_dL_db3
        b3_new = b3 - learn_rate*momn_dL_db3
        prev_dL_db3 = momn_dL_db3
        
        dL_db2 = (1./batch_size)*np.sum(np.matmul(W3.T,p-y_batch)*sigmoid_prime(q2), axis=1, keepdims=True)
        momn_dL_db2 = dL_db2 + beta*prev_dL_db2
        b2_new = b2 - learn_rate*momn_dL_db2
        prev_dL_db2 = momn_dL_db2
        
        dL_db1 = (1./batch_size)*np.sum(np.matmul(W2.T,np.matmul(W3.T, p-y_batch))*sigmoid_prime(q2)*sigmoid_prime(q1), axis=1, keepdims=True)
        momn_dL_db1 = dL_db1 + beta*prev_dL_db1
        b1_new = b1 - learn_rate*momn_dL_db1
        prev_dL_db1 = momn_dL_db1
        
        # Updtae the weights and biases to the new values
        W1 = W1_new
        W2 = W2_new
        W3 = W3_new
        b1 = b1_new
        b2 = b2_new
        b3 = b3_new
        
        #store the final p
        p_final[:, b_no*32: (b_no*32 + 32)] = p

    #== Training error ===========================================================
    # Cross-entropy Error
    err_train = 0    
    for i in range(N_train):
        err_train = err_train + np.dot(y[:,i].T, np.log(1/p_final[:,i]))
    avg_err_train += [err_train/N_train];
    # Classification error
    no_err_train = 0
    for i in range(N_train):
        max_p_pos = np.argmax(p_final[:,i])
        if y[max_p_pos,i] != 1:
            no_err_train += 1
    avg_class_err_train += [100*no_err_train/N_train]
    
    #== VALIDATION ==============================================================
    #   on the whole 20000 dataset
    q1_val = np.matmul(W1,x_val) + b1
    h1_val = sigmoid(q1_val)
    q2_val = np.matmul(W2,h1_val) + b2
    h2_val = sigmoid(q2_val)
    o_val = np.matmul(W3,h2_val) + b3
    p_val = softmax(o_val)  # - Predicted label        
        
    # Cross-entropy error
    err_val = 0    
    for i in range(N_val):
        err_val = err_val + np.dot(y_val[:,i].T, np.log(1/p_val[:,i]))
    avg_err_val += [err_val/N_val];
    # Classification error
    no_err_val = 0
    for i in range(N_val):
        max_p_pos = np.argmax(p_val[:,i])
        if y_val[max_p_pos,i] != 1:
            no_err_val += 1
    avg_class_err_val += [100*no_err_val/N_val]
    #== Testing ==============================================================
    q1_test = np.matmul(W1,x_test) + b1
    h1_test = sigmoid(q1_test)
    q2_test = np.matmul(W2,h1_test) + b2
    h2_test = sigmoid(q2_test)
    o_test = np.matmul(W3,h2_test) + b3
    p_test = softmax(o_test)  # - Predicted label        
        
    # Cross-entropy error
    err_test = 0    
    for i in range(N_test):
        err_test += np.dot(y_test[:,i].T, np.log(1/p_test[:,i]))
    avg_err_test += [err_test/N_test];
    # Classification error
    no_err_test = 0
    for i in range(N_test):
        max_p_pos = np.argmax(p_test[:,i])
        if y_test[max_p_pos,i] != 1:
            no_err_test += 1
    avg_class_err_test += [100*no_err_test/N_test]
    #==============================================================================
    
    print('\nEpoch = ', epoch)
    print('Train Loss = ', avg_err_train[epoch])
    print('Val Loss = ', avg_err_val[epoch])
    print('Test Loss = ', avg_err_test[epoch])
    
    print('Train error = ', avg_class_err_train[epoch])
    print('Val error = ', avg_class_err_val[epoch])
    print('Test error = ', avg_class_err_test[epoch])


# Plot Cross Entropy Loss
plt.figure()  
plt.plot(range(0,no_epoch), avg_err_train, label = 'Training Loss')
plt.plot(range(0,no_epoch), avg_err_val, c = 'r', label = 'Validation Loss')
plt.plot(range(0,no_epoch), avg_err_test, c = 'g', label = 'Testing Loss')
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
        
## Visualize weights
#plt.subplots(10,10)
#for i in range(100):
#    weight = W1[i,:]              
#    image = np.reshape(weight, (28,56))
#    plt.subplot(10,10,i+1)
#    plt.imshow(image, cmap = 'gray')
#    
#plt.show()