import numpy as np
import matplotlib.pyplot as plt
import csv

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(o):
    temp = np.exp(o)
    (r,c) = np.shape(o)
    for i in range(c):
        sum = np.sum(temp[:,i])
        temp[:,i] = temp[:,i]/sum
    
    return temp

#def sigmoid_prime(x):
#    return sigmoid(x)*(1-sigmoid(x))

def sigmoid_prime(x):
    return np.exp(-x)/((1+np.exp(-x))**2)

# Input Matrix, 1568 x 20000
N_train = 20000 
input_x = np.zeros((1568,N_train))
# Labels
y = np.zeros((19,N_train))

# import validation data
N_val = 5000
input_x_val = np.zeros((1568,N_val))
# Labels
y_val = np.zeros((19,N_val))
count = 0
with open('val.txt') as csv_file:
    data_csv = csv.reader(csv_file, delimiter = ',')
    for data in data_csv:
        sum = [data[1568]]
        sum = np.array(sum, dtype = 'int')
        y_val[sum,count] = 1
        
        pixels = data[0:1568]              
        # convert string data type list into floating data type array
        image_vec = np.array(pixels, dtype = 'float')
        input_x_val[:,count] = image_vec
        count = count + 1
#        print(count)

# Import test data
N_test = 5000
input_x_test = np.zeros((1568,N_test))
# Labels
y_test = np.zeros((19,N_val))
count = 0
with open('test.txt') as csv_file:
    data_csv = csv.reader(csv_file, delimiter = ',')
    for data in data_csv:
        sum = [data[1568]]
        sum = np.array(sum, dtype = 'int')
        y_test[sum,count] = 1
        
        pixels = data[0:1568]              
        # convert string data type list into floating data type array
        image_vec = np.array(pixels, dtype = 'float')
        input_x_test[:,count] = image_vec
        count = count + 1
#        print(count)





# Number of Epochs
no_epoch = 30
# learning rate
learn_rate = 0.5
# Momentum
beta = 0.5
# Number of hidden units
H = 200

# W should be uniform between [-u,u]
u = np.sqrt(6/H)
np.random.seed(1)
W1 = np.random.uniform(-u,u,(H,1568))
W2 = np.random.uniform(-u,u, (19,H))
# b should be initialised to 0
b1 = np.zeros((H,1))
b2 = np.zeros((19,1))

# Predicted output (or the probability) 19x2000
p_final = np.zeros((19,N_train))

prev_dL_dW2, prev_dL_dW1, prev_dL_db2, prev_dL_db1 = 0,0,0,0

avg_err_train, avg_err_val, avg_err_test = [], [], []
avg_class_err_train, avg_class_err_val, avg_class_err_test = [], [], []
for epoch in range(no_epoch):  
# Mini batch gradient descent
    # no of batch = 20000/32 = 625
    batch_size = 32
    no_batches = 625
    for b_no in range(no_batches):
        input_x_batch = input_x[:, b_no*32: (b_no*32 + 32)]
        y_batch = y[:, b_no*32: (b_no*32 + 32)]
    #== TRAINING =============================================================
        q = np.matmul(W1,input_x_batch) + b1
        h = sigmoid(q)
        o = np.matmul(W2,h) + b2
        p = softmax(o)  
    #=========================================================================
    #   Updating dL_dW2 ------------------------------------------------------
        dL_dW2 = (1./batch_size)*np.multiply(1,np.matmul((p-y_batch),h.T))
        momn_dL_dW2 = dL_dW2 + beta*prev_dL_dW2
        W2_new = W2 - learn_rate*momn_dL_dW2
        prev_dL_dW2 = momn_dL_dW2
        
        dL_dW1 = (1./batch_size)*np.multiply(1,np.matmul(np.matmul(W2.T, p-y_batch)*sigmoid_prime(q),input_x_batch.T))
        momn_dL_dW1 = dL_dW1 + beta*prev_dL_dW1
        W1_new = W1 - learn_rate*momn_dL_dW1
        prev_dL_dW1 = momn_dL_dW1
        
    #   Updating dL_db2
        dL_db2 = (1./batch_size)*np.sum(p-y_batch, axis=1, keepdims=True)
        momn_dL_db2 = dL_db2 + beta*prev_dL_db2
        b2_new = b2 - learn_rate*momn_dL_db2
        prev_dL_db2 = momn_dL_db2
        
    #   Updating dL_db1
    #   dL_db1 = np.matmul(W2.T,p-y)*sigmoid_prime(q)
        dL_db1 = (1./batch_size)*np.sum(np.matmul(W2.T,p-y_batch)*sigmoid_prime(q), axis=1, keepdims=True)
        momn_dL_db1 = dL_db1 + beta*prev_dL_db1
        b1_new = b1 - learn_rate*momn_dL_db1
        prev_dL_db1 = momn_dL_db1
        
        # Updtae the weights and biases to the new values
        W1 = W1_new
        W2 = W2_new
        b1 = b1_new
        b2 = b2_new  
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
    q_val = np.matmul(W1,input_x_val) + b1
    h_val = sigmoid(q_val)
    o_val = np.matmul(W2,h_val) + b2
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
#==============================================================================

#== Testing ==============================================================
#   on the whole 20000 dataset
    q_test = np.matmul(W1,input_x_test) + b1
    h_test = sigmoid(q_test)
    o_test = np.matmul(W2,h_test) + b2
    p_test = softmax(o_test)  # - Predicted label
       
    # Cross-entropy error
    err_test = 0    
    for i in range(N_test):
        err_test = err_test + np.dot(y_test[:,i].T, np.log(1/p_test[:,i]))
    avg_err_test += [err_test/N_test];
    # Classification error
    no_err_test = 0
    for i in range(N_test):
        max_p_pos = np.argmax(p_test[:,i])
        if y_test[max_p_pos,i] != 1:
            no_err_test += 1
    avg_class_err_test += [100*no_err_test/N_test]
#==============================================================================
    
    print('Epoch = ', epoch)
    print('Avg. Train Error = ', err_train/N_train,  '\n Avg. Val Error = ', err_val/N_val, '\n Avg. test Error = ', err_test/N_test)


# Plot Cross Entropy Error 
plt.figure()  
plt.plot(range(0,no_epoch), avg_err_train, label = 'Training error')
plt.plot(range(0,no_epoch), avg_err_val, c = 'r', label = 'Validation error')
plt.plot(range(0,no_epoch), avg_err_test, c = 'g', label = 'Testing error')
plt.legend(loc = 'upper right')
plt.xlabel('Epoch')
plt.ylabel('Cross-entropy Error')

# Plot Classification Error 
plt.figure()   
plt.plot(range(0,no_epoch), avg_class_err_train, label = 'Training error')
plt.plot(range(0,no_epoch), avg_class_err_val, c = 'r', label = 'Validation error')
plt.plot(range(0,no_epoch), avg_class_err_test, c = 'g', label = 'Testing error')
plt.legend(loc = 'upper right')
plt.xlabel('Epoch')
plt.ylabel('Percentage Classification Error')
       
# Visualize weights
plt.subplots(10,10)
for i in range(100):
    weight = W1[i,:]              
    image = np.reshape(weight, (28,56))
    plt.subplot(10,10,i+1)
    plt.imshow(image, cmap = 'gray')
    
plt.show()