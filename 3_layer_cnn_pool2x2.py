import numpy as np
import matplotlib.pyplot as plt

def softmax(o):
    temp = np.exp(o) 
    sum_col = np.sum(temp, axis=0)
    return temp/sum_col
#--------------------------------------------------------------------------- 
def get_im2col_ind(x_shape, field_height, field_width, padding, stride):
   N, C, H, W = x_shape
   assert (H + 2 * padding - field_height) % stride == 0
   assert (W + 2 * padding - field_height) % stride == 0
   out_height = int((H + 2 * padding - field_height) / stride + 1)
   out_width = int((W + 2 * padding - field_width) / stride + 1)

   i0 = np.repeat(np.arange(field_height), field_width)
   i0 = np.tile(i0, C)
   i1 = stride * np.repeat(np.arange(out_height), out_width)
   j0 = np.tile(np.arange(field_width), field_height * C)
   j1 = stride * np.tile(np.arange(out_width), out_height)
   i = i0.reshape(-1, 1) + i1.reshape(1, -1)
   j = j0.reshape(-1, 1) + j1.reshape(1, -1)

   k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

   return (k.astype(int), i.astype(int), j.astype(int))
#--------------------------------------------------------------------------- 
def im2col_ind(x, field_height, field_width, padding, stride):
   p = padding
   x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

   k, i, j = get_im2col_ind(x.shape, field_height, field_width, padding, stride)

   cols = x_padded[:, k, i, j]
   C = x.shape[1]
   cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
   return cols
#--------------------------------------------------------------------------- 
def col2im_indices(cols, x_shape, field_height, field_width, padding, stride):
   N, C, H, W = x_shape
   H_padded, W_padded = H + 2 * padding, W + 2 * padding
   x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
   k, i, j = get_im2col_ind(x_shape, field_height, field_width, padding, stride)
   cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
   cols_reshaped = cols_reshaped.transpose(2, 0, 1)
   np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
   if padding == 0:
       return x_padded
   return x_padded[:, :, padding:-padding, padding:-padding]
#--------------------------------------------------------------------------- 
def conv_forward(X, W, padding, stride):
    store = W, stride, padding
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    h_out, w_out = int(h_out), int(w_out)
    X_col = im2col_ind(X, h_filter, w_filter, padding, stride)
    W_col = W.reshape(n_filters, -1)
    
    out = np.matmul(W_col,X_col)
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)
    store = (X, W, stride, padding, X_col)

    return out, store
#--------------------------------------------------------------------------- 
def conv_backward(dout, store):
    X, W, stride, padding, X_col = store
    n_filter, d_filter, h_filter, w_filter = W.shape
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)

    dW = np.matmul(dout_reshaped, X_col.T)
    dW = dW.reshape(W.shape)
    return dW
#--------------------------------------------------------------------------- 
#def pooling(h,p):
##    pooling filter = pxp = 2x2
##    Input dimensions: h = n_filters x n x n = 36x32x32
#    N_images,n_filters,n,n = np.shape(h)
#    m = 32//p     # gives an integer after division instead of float
##    Output dimesnsion: g = n_filters x m x m = 36x16x16
#    pool_out = np.zeros((N_images,n_filters,m,m))
#    for img in range(N_images):
#        for k in range(n_filters):
#            for i in range(0,m,p):
#                for j in range(0,m,p):
#                    pool_out[img,k,i//p,j//p] = np.max(h[img,k,i:i+p,j:j+p])
#    return pool_out
#---------------------------------------------------------------------------   
def pool_forward(X, p, padding, stride):
    n, d, h, w = X.shape
    h_filter, w_filter = p, p
    h_out = (h - h_filter + 2 * padding) / stride + 1
    w_out = (w - w_filter + 2 * padding) / stride + 1
    h_out, w_out = int(h_out), int(w_out)
    
    X_reshaped = X.reshape(n * d, 1, h, w)
    X_col = im2col_ind(X_reshaped, p, p, padding, stride)
    max_idx = np.argmax(X_col, axis=0)
    
    out = X_col[max_idx, range(max_idx.size)]
    out = out.reshape(h_out, w_out, n, d)
    out = out.transpose(2, 3, 0, 1)
    return out, X_col, max_idx
#---------------------------------------------------------------------------  
def pool_backward(dout, X_col, max_idx, h_dim, p, padding, stride):
    n, d, h, w = h_dim
    dX_col = np.zeros_like(X_col)    
    dout_flat = dout.transpose(2, 3, 0, 1).ravel()
    dX_col[max_idx, range(max_idx.size)] = dout_flat

    dX = col2im_indices(dX_col, (n * d, 1, h, w), p, p, padding, stride)
    dX = dX.reshape(h_dim)
    return dX
#--------------------------------------------------------------------------- 
# number of classes
n_class = 10

# Input Matrix, 1568 x 20000
N_total = 20000
N_train = 10240
N_val = 5000
N_test = 4760

#x_train = flat_Xn[0:N_train,:]
#x_train= x_train.T

all_image = np.zeros((N_train,1,50,50)) 
all_image[:,0,:,:] = Xn[0:N_train,...]

all_image_val  = np.zeros((N_val,1,50,50)) 
all_image_val[:,0,:,:] = Xn[N_train:N_train+N_val,...]

all_image_test = np.zeros((N_test,1,50,50)) 
all_image_test[:,0,:,:] = Xn[N_train+N_val:N_total,...] 
#x_val = flat_Xn[N_train:N_train+N_val,:]
#x_val = x_val.T
#x_test = flat_Xn[N_train+N_val:N_total,:]
#x_test = x_test.T

# One-hot encoding
y_hot = np.zeros((n_class,N_total))
for i in range(N_total):
    y_hot[y_label[0,i],i] = 1

y = y_hot[:,0:N_train]
train_label_hot = y
y_val = y_hot[:,N_train:N_train+N_val]
val_label_hot = y_val
y_test = y_hot[:,N_train+N_val:N_total]
test_label_hot = y_test

# Input data and labels
#train_x = train['data']
#train_labels = train['labels']
#train_labels = np.array([train_labels]).T
## Shuffling
#temp1 = np.append(train_x,train_labels, axis = 1) 
#np.random.shuffle(temp1)
#train_x = temp1[:,0:3072]
#train_labels = temp1[:,3072]
##N_train = len(train_labels)
#N_train = 9984
## 9984
## One hot encoding
#train_label_hot = np.zeros((n_class,N_train))
#for i in range(N_train):
#    train_label_hot[int(train_labels[i]),i] = 1
#
## Reshape into RGB image
#all_red = train_x[0:N_train,0:1024]
#all_green = train_x[0:N_train,1024:2048]
#all_blue = train_x[0:N_train,2048:3072]
#
#all_red_channel = np.reshape(all_red, (N_train,32,32)) 
#all_green_channel = np.reshape(all_green, (N_train,32,32))
#all_blue_channel = np.reshape(all_blue, (N_train,32,32))
#
#all_image = np.zeros((N_train,3,32,32)) 
#all_image[:,0,:,:] = all_red_channel
#all_image[:,1,:,:] = all_green_channel
#all_image[:,2,:,:] = all_blue_channel
# Plot the image
#plot_image(train_x[0,:])

#-----------------------------------------------------------------------------
#val_x = val['data']
#val_labels = val['labels']
#val_labels = np.array([val_labels]).T
#N_val = len(val_labels)
## One hot encoding
#val_label_hot = np.zeros((n_class,N_val))
#for i in range(N_val):
#    val_label_hot[int(val_labels[i]),i] = 1
#
## Reshape into RGB image
#all_red_val = val_x[0:N_val,0:1024]
#all_green_val = val_x[0:N_val,1024:2048]
#all_blue_val = val_x[0:N_val,2048:3072]
#
#all_red_channel_val = np.reshape(all_red_val, (N_val,32,32)) 
#all_green_channel_val = np.reshape(all_green_val, (N_val,32,32))
#all_blue_channel_val = np.reshape(all_blue_val, (N_val,32,32))
#
#all_image_val  = np.zeros((N_val,3,32,32)) 
#all_image_val [:,0,:,:] = all_red_channel_val 
#all_image_val [:,1,:,:] = all_green_channel_val 
#all_image_val [:,2,:,:] = all_blue_channel_val   


#------------------------------------------------------------------------------
#test_x = test['data']
#test_labels = test['labels']
#test_labels = np.array([test_labels]).T
#N_test = len(test_labels)
## One hot encoding
#test_label_hot = np.zeros((n_class,N_test))
#for i in range(N_val):
#    test_label_hot[int(test_labels[i]),i] = 1
#
## Reshape into RGB image
#all_red_test = test_x[0:N_test,0:1024]
#all_green_test = test_x[0:N_test,1024:2048]
#all_blue_test = test_x[0:N_test,2048:3072]
#
#all_red_channel_test = np.reshape(all_red_test, (N_val,32,32)) 
#all_green_channel_test = np.reshape(all_green_test, (N_val,32,32))
#all_blue_channel_test = np.reshape(all_blue_test, (N_val,32,32))
#
#all_image_test  = np.zeros((N_test,3,32,32)) 
#all_image_test [:,0,:,:] = all_red_channel_test 
#all_image_test [:,1,:,:] = all_green_channel_test 
#all_image_test [:,2,:,:] = all_blue_channel_test   

#------------------------------------------------------------------------------  
# filter size = n_filters x 3 x k x k
n_filters, rgb, k = 5, 1, 5
# Number of Epochs
no_epoch, learn_rate, beta , wd_lambda = 10, 0.01, 0, 0
# Momentum, # Weight decay
# Number of hidden units
H = 36

np.random.seed(1)
u = np.sqrt(6/(H))
var = 1/(n_filters*rgb*k*k)
# W uniform between [-u,u]
Wcnn = np.random.normal(0,0.15,(n_filters,rgb,k,k))
W1 = np.random.uniform(-u,u,(H,n_filters*25*25))
W2 = np.random.uniform(-u,u,(n_class,H))
#W2 = np.random.uniform(-u,u,(n_class,n_filters*16*16))
# b should be initialised to 0
b1 = np.zeros((H,1))
b2 = np.zeros((n_class,1))

# no of batch = 20000/32 = 625
batch_size = 32
no_batches = int(N_train/batch_size)
    
# Predicted output (or the probability) 19x2000
pred = np.zeros((n_class,N_train))
pred_val = np.zeros((n_class,N_val))
pred_test = np.zeros((n_class,N_test))

prev_dWcnn, prev_dL_dW1, prev_dL_dW2, prev_dL_db1, prev_dL_db2 = 0, 0, 0, 0, 0
avg_ent_loss, avg_ent_loss_val = [], []
avg_class_err_train, avg_class_err_val, avg_class_err_test = [], [], []
for epoch in range(no_epoch):
    # Mini batch gradient descent
    for b_no in range(no_batches):
        X_batch = all_image[b_no*batch_size: (b_no+1)*batch_size,:,:,:]
        y_batch = train_label_hot[:,b_no*batch_size: (b_no+1)*batch_size]
        
        # Convolution
        padding,stride = 2, 1
        conv, store = conv_forward(X_batch, Wcnn, padding, stride)
        # ReLU activation 
        h_conv_out = np.maximum(conv,0) 
            
        # Pooling by pxp
        pool_size, pool_padding, pool_stride = 2, 0, 2
        g_pool_out, X_col_pool, max_idx = pool_forward(h_conv_out, pool_size, 
                                                    pool_padding, pool_stride)
        batch_size,n_filters,m,m = g_pool_out.shape       
        # To fully connected layer
        g_pool_flat = np.reshape(g_pool_out,(batch_size,n_filters*m*m))
        #=====================================================================        
        input_X_batch = g_pool_flat.T
        q = np.matmul(W1,input_X_batch) + b1
        h = np.maximum(q,0)
        o = np.matmul(W2,h) + b2
        p = softmax(o)				
        #store prediction 'p' from this batch to final 10xN_train prediction matrix
        pred[:, b_no*batch_size: (b_no+1)*batch_size] = p

    #   Updating dL_dW2 ------------------------------------------------------
        dL_dW2 = (1./batch_size)*np.multiply(1,np.matmul((p-y_batch),h.T))
        momn_dL_dW2 = dL_dW2 + beta*prev_dL_dW2
        W2_new = W2 - learn_rate*momn_dL_dW2
        prev_dL_dW2 = momn_dL_dW2
		
		# ReLU indices
        relu = np.ones((h.shape))
        temp = np.argwhere(h == 0)
        for i in range(len(temp)):
            relu[temp[i,0],temp[i,1]] = 0
        dL_dW1 = (1./batch_size)*np.multiply(1,np.matmul(
                            np.matmul(W2.T, p-y_batch)*relu,input_X_batch.T))
        momn_dL_dW1 = dL_dW1 + beta*prev_dL_dW1
        W1_new = W1 - learn_rate*momn_dL_dW1
        prev_dL_dW1 = momn_dL_dW1       

        dL_db2 = (1./batch_size)*np.sum(p-y_batch, axis=1, keepdims=True)
        momn_dL_db2 = dL_db2 + beta*prev_dL_db2
        b2_new = b2 - learn_rate*momn_dL_db2
        prev_dL_db2 = momn_dL_db2        

        dL_db1 = (1./batch_size)*np.sum(np.matmul(W2.T,p-y_batch)*
                                     relu, axis=1, keepdims=True)
        momn_dL_db1 = dL_db1 + beta*prev_dL_db1
        b1_new = b1 - learn_rate*momn_dL_db1
        prev_dL_db1 = momn_dL_db1    
        
        dX = np.matmul(W1.T, np.matmul(W2.T, p-y_batch)*relu)
        dg_reshaped = np.reshape(dX.T, (batch_size,n_filters,m,m))
        h_dim = h_conv_out.shape
        dh = pool_backward(dg_reshaped, X_col_pool, max_idx, h_dim,
                                       pool_size, pool_padding, pool_stride)
        # find the zeros in h
        temp = np.argwhere(h_conv_out == 0)
        for i in range(len(temp)):
            dh[temp[i,0],temp[i,1],temp[i,2],temp[i,3]] = 0
            
        dWcnn = (1./batch_size)*conv_backward(dh, store)
#        print(np.linalg.norm(dWcnn))
        momn_dWcnn= dWcnn + beta*prev_dWcnn
        Wcnn_new = Wcnn - learn_rate*momn_dWcnn
        prev_dWcnn = momn_dWcnn       
                         
        W1 = W1_new
        W2 = W2_new
        b1 = b1_new
        b2 = b2_new        
        Wcnn = Wcnn_new
        
        print('Epoch=',epoch, 'Batch=',b_no)
        
    #== Training Loss and Error  ===============================================
    # Cross-entropy loss
    avg_ent_loss += [(1./N_train)*np.trace(np.matmul(train_label_hot.T, np.log(1/pred)))]
    print('Epoch = ',epoch, 'Training loss =',avg_ent_loss[epoch])
    # Training error ---------------------------------------------------------
    no_err_train = 0
    for i in range(N_train):
        max_p_pos = np.argmax(pred[:,i])
        if train_label_hot[max_p_pos,i] != 1:
            no_err_train += 1
    avg_class_err_train += [100*no_err_train/N_train]
    print('Epoch = ',epoch, 'Training error =',avg_class_err_train[epoch])
           
    #== VALIDATION ============================================================ 
    #total N_val = 5000 images. Each batch = 100 images. Total 50 batches
    batch_size_val = 100
    for b_no_val in range(50):       
        X_batch_val = all_image_val[b_no_val*batch_size_val: (b_no_val+1)*batch_size_val,:,:,:]        
        # Convolution
        conv_val, store_val = conv_forward(X_batch_val, Wcnn, padding, stride)
        # ReLU activation 
        h_conv_out_val = np.maximum(conv_val,0)             
        # Pooling by pxp
        g_pool_out_val, X_col_pool_val, max_idx_val = pool_forward(h_conv_out_val, pool_size, 
                                                    pool_padding, pool_stride)
        batch_size_val,n_filters,m,m = g_pool_out_val.shape       
        # To fully connected layer
        g_pool_flat_val = np.reshape(g_pool_out_val,(batch_size_val,n_filters*m*m))
        #=====================================================================        
        input_X_batch_val = g_pool_flat_val.T
        q_val = np.matmul(W1,input_X_batch_val) + b1
        h_val = np.maximum(q_val,0)		
        o_val = np.matmul(W2,h_val) + b2
        p_val = softmax(o_val)       
        pred_val[:, b_no_val*batch_size_val: (b_no_val+1)*batch_size_val] = p_val
    # Validation loss
    avg_ent_loss_val += [(1./N_val)*np.trace(np.matmul(val_label_hot.T, np.log(1/pred_val)))]
    print('Epoch = ',epoch, 'Validation loss =', avg_ent_loss_val[epoch])              
    # Validation Error
    no_err_val = 0
    for i in range(N_val):
        max_p_pos = np.argmax(pred_val[:,i])
        if val_label_hot[max_p_pos,i] != 1:
            no_err_val += 1
    avg_class_err_val += [100*no_err_val/N_val]
    print('Epoch = ',epoch, 'Validation Error =', avg_class_err_val[epoch])                   
    #== Testing ==============================================================
    #total N_test = 5000 images. Each batch = 100 images. Total 50 batches
    batch_size_test = 100
    for b_no_test in range(50):  
        X_batch_test = all_image_test[b_no_test*batch_size_test: (b_no_test+1)*batch_size_test,:,:,:]        
            # Convolution
        conv_test, store_test = conv_forward(X_batch_test, Wcnn, padding, stride)
            # ReLU activation 
        h_conv_out_test = np.maximum(conv_test,0)             
            # Pooling by pxp
        g_pool_out_test, X_col_pool_test, max_idx_test = pool_forward(h_conv_out_test, pool_size, 
                                                        pool_padding, pool_stride)
        batch_size_test,n_filters,m,m = g_pool_out_test.shape       
            # To fully connected layer
        g_pool_flat_test = np.reshape(g_pool_out_test,(batch_size_test,n_filters*m*m))
                  
        input_X_batch_test = g_pool_flat_test.T
        q_test = np.matmul(W1,input_X_batch_test) + b1
        h_test = np.maximum(q_test,0)		
        o_test = np.matmul(W2,h_test) + b2
        p_test = softmax(o_test)      
        pred_test[:, b_no_test*batch_size_test: (b_no_test+1)*batch_size_test] = p_test                
    # Testing Error
    no_err_test = 0
    for i in range(N_test):
        max_p_pos = np.argmax(pred_test[:,i])
        if test_label_hot[max_p_pos,i] != 1:
            no_err_test += 1
    avg_class_err_test += [100*no_err_test/N_test]
    print('Testing Error =', avg_class_err_test)  
#------------------------------------------------------------------------------     




