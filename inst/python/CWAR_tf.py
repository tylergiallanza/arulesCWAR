import numpy as np
import scipy
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

def r_to_py_scipy_sparse(i,p):
    data = [1 for _ in range(len(i))]
    csc_mat = scipy.sparse.csc_matrix((data,i,p))
    #coo_mat = scipy.sparse.csc_matrix.tocoo(csc_mat)
    csr_mat = scipy.sparse.csc_matrix.tocsr(csc_mat)
    return csr_mat

def get_x_batch(csr_mat, start_index, batch_size):
    batched_mat = csr_mat[start_index:start_index+batch_size]
    coo_mat = scipy.sparse.csr_matrix.tocoo(batched_mat)
    return tf.compat.v1.SparseTensorValue(indices=np.array([coo_mat.row, coo_mat.col]).T,
      values=coo_mat.data,
      #dense_shape=(batch_size,coo_mat.shape[1]))
      dense_shape=coo_mat.shape)

# used for training
def build_arch(num_rules, num_classes, deep, loss, optimizer, opt_params, l1, l2):
    tf.compat.v1.reset_default_graph()
    y = tf.compat.v1.placeholder(tf.compat.v1.float32, (None, num_classes))
    x = tf.compat.v1.sparse_placeholder(tf.compat.v1.float32, (None, num_rules))
    
    # use grouping layer of size deep?
    if deep > 0:
        w1 = tf.compat.v1.get_variable('w1',initializer=tf.compat.v1.truncated_normal_initializer(),shape=(num_rules,deep))
        b1 = tf.compat.v1.Variable(tf.compat.v1.zeros((deep,)))
        w2 = tf.compat.v1.get_variable('w2',initializer=tf.compat.v1.truncated_normal_initializer(),shape=(deep,num_classes))
        b2 = tf.compat.v1.Variable(tf.compat.v1.zeros((num_classes,)))
    else:
        w1 = tf.compat.v1.get_variable('w1',initializer=tf.compat.v1.truncated_normal_initializer(),shape=(num_rules,num_classes))
        b1 = tf.compat.v1.Variable(tf.compat.v1.zeros((num_classes,)))
        w2, b2 = 0, 0
    
    # make sure w1 is non-negative
    w1 = tf.compat.v1.nn.relu(w1)
    
    # cross product plus bias
    first_out = tf.compat.v1.add(tf.compat.v1.sparse_tensor_dense_matmul(x, w1), b1)
    
    if deep:
        first_out = tf.compat.v1.nn.relu(first_out) # output needs to be positive???
        output = tf.compat.v1.add(tf.compat.v1.matmul(first_out, w2, a_is_sparse = True, b_is_sparse = True), b2)
    else:
        output = first_out
    
    yhat = tf.compat.v1.nn.softmax(output)
    
    if loss=='mse':
        loss = tf.compat.v1.losses.mean_squared_error(y,yhat)
    elif loss == 'cross':
        loss = tf.compat.v1.losses.softmax_cross_entropy(y,yhat)
    
    regularization = 0.0
    if l1:
        regularization = tf.compat.v1.add(regularization, tf.compat.v1.scalar_mul(l1, tf.compat.v1.reduce_sum(w1)))
    if l2:
        regularization = tf.compat.v1.add(regularization, tf.compat.v1.scalar_mul(l2, tf.compat.v1.reduce_sum(tf.compat.v1.square(w1))))
    
    loss = tf.compat.v1.add(loss, regularization)
    
    if optimizer == 'sgd':
        if type(opt_params) is list:
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(*opt_params)
        else:
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(opt_params)
    elif optimizer == 'adam':
        optimizer = tf.compat.v1.train.AdamOptimizer(*opt_params)
    elif optimizer == 'adadelta':
        optimizer = tf.compat.v1.train.AdadeltaOptimizer(*opt_params)
        
    train_step = optimizer.minimize(loss)
    prediction = tf.compat.v1.argmax(yhat,1)
    correct_prediction = tf.compat.v1.equal(prediction,tf.compat.v1.argmax(y,1))
    rules = tf.compat.v1.count_nonzero(tf.compat.v1.reduce_sum(tf.compat.v1.nn.relu(w1),1))
    accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_prediction,tf.compat.v1.float32))
    
    return {'x':x, 'y':y, 
            'train_step':train_step, 'prediction':prediction, 
            'loss':loss, 'accuracy':accuracy, 'rules':rules, 
            'w1':w1, 'b1':b1, 'w2':w2, 'b2':b2}

def train(tensors, epochs, batch_size, x_data, y_data, deep, prod=False, 
          x_test = None, y_test = None, x_val = None, y_val = None):
    indices = list(range(len(y_data)))
    
    with tf.compat.v1.Session() as sess:
        last_acc, last_num_rules = 0, 0
        batch_size = min(batch_size, len(indices))
        sess.run(tf.compat.v1.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss, epoch_acc = 0, 0
            num_batches = (len(indices) - len(indices) % batch_size) / batch_size
            for start_index in range(0,len(indices) - len(indices) % batch_size, batch_size):
                #x_data is not batch subsetting correctly - try using tf slice
                # MFH: was the problem the () in the range?
                batch_x, batch_y = get_x_batch(x_data, start_index, batch_size), y_data[start_index:(start_index+batch_size)]
                feed_dict = {tensors['x']:batch_x, tensors['y']:batch_y}
                _, loss,acc,num_rules = sess.run([tensors['train_step'], tensors['loss'], 
                                                  tensors['accuracy'], tensors['rules']], feed_dict)
                epoch_loss += loss
                epoch_acc += acc
                last_acc, last_num_rules = acc, num_rules
            epoch_loss  /= num_batches
            epoch_acc /= num_batches
            if not prod and False:
                print(epoch_loss,epoch_acc,num_rules)
        if prod:
            test_acc, val_acc = None, None
            if x_test is not None:
                batch_x = get_x_batch(x_test, 0, len(y_test))
                feed_dict = {tensors['x']:batch_x, tensors['y']:y_test}
                test_acc = sess.run(tensors['accuracy'], feed_dict) 
            if x_val is not None:
                batch_x = get_x_batch(x_val, 0, len(y_val))
                feed_dict = {tensors['x']:batch_x, tensors['y']:y_val}
                val_acc = sess.run(tensors['accuracy'], feed_dict) 
            return last_acc, test_acc, val_acc, last_num_rules
        else:
            weights = {}
            weights['w1'] = sess.run(tensors['w1'])
            weights['b1'] = sess.run(tensors['b1'])
            if deep:
                weights['w2'] = sess.run(tensors['w2'])
                weights['b2'] = sess.run(tensors['b2'])
            
            return weights

### The rest of the file is currently UNUSED!

# used for predict
def build_arch_for_object(num_rules, w1_val, w2_val, b1_val, b2_val):
    tf.compat.v1.reset_default_graph()
    x = tf.compat.v1.sparse_placeholder(tf.compat.v1.float64, (None, num_rules))
    w1 = tf.compat.v1.Variable(w1_val)
    b1 = tf.compat.v1.Variable(b1_val)
    w2 = tf.compat.v1.Variable(w2_val)
    b2 = tf.compat.v1.Variable(b2_val)
    w1 = tf.compat.v1.nn.relu(w1)
    first_out = tf.compat.v1.add(tf.compat.v1.sparse_tensor_dense_matmul(x,w1),b1)
    first_out = tf.compat.v1.nn.relu(first_out)
    output = tf.compat.v1.add(tf.compat.v1.matmul(first_out, w2, a_is_sparse=True, b_is_sparse=True), b2)
    yhat = tf.compat.v1.nn.softmax(output)
    return {'yhat':yhat,'x':x,'first_out':first_out}

def eval_tensors(tensors, name, data):
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        if name == 'yhat':
            return np.argmax(sess.run(tensors[name], 
                                      feed_dict = {tensors['x']:get_x_batch(data, 0, data.shape[0])}), 1)
        else:
            return sess.run(tensors[name], feed_dict={tensors['x']:get_x_batch(data, 0, data.shape[0])})


# the following us currently not used
def nested_crossval_fold(train_param_choices, opt_param_choices, x_train, y_train, x_test, y_test, x_val, y_val):
    model_data = []
    for train_params in train_param_choices:
        for opt_params in opt_param_choices:
            current_arch = build_arch(x_train.shape[1], y_train.shape[1], train_params['deep'], train_params['loss'], train_params['optimizer'], opt_params, train_params['l1'], train_params['l2'])
            train_acc, test_acc, val_acc, current_nr = train(current_arch, train_params['epochs'], train_params['batch_size'], x_train, y_train, True, x_test, y_test, x_val, y_val)
            model_data.append({})
            model_data[-1]['num_rules'] = current_nr
            model_data[-1]['train_acc'] = train_acc
            #model_data[-1]['test_acc'] = test_acc
            model_data[-1]['val_acc'] = val_acc
            model_data[-1]['train_params'] = train_params
            model_data[-1]['opt_params'] = opt_params
    return model_data
        
def pick_best_model(model_history):
    model_averages = [0 for _ in range(len(model_history[0]))]
    for models in model_history:
        for model_index,model in enumerate(models):
            model_averages[model_index] += model['val_acc']
    best_idx = np.array(model_averages).argmax()
    return model_history[0][best_idx]
