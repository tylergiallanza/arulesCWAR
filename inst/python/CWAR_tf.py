import numpy as np
import scipy
import tensorflow as tf
import sys

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
def build_arch(num_rules, num_classes, deep, loss, optimizer, opt_params, l1, l2, l1_path = False, l2_path = False):
    if l1_path and l2_path:
        print('Error - only one regularization path is supported.')
        return None
    tf.compat.v1.reset_default_graph()
    y = tf.compat.v1.placeholder(tf.compat.v1.float32, (None, num_classes))
    x = tf.compat.v1.sparse_placeholder(tf.compat.v1.float32, (None, num_rules))
    reg_in = tf.compat.v1.placeholder(tf.compat.v1.float32,())
    
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
        #first_out = tf.compat.v1.nn.relu(first_out) # output needs to be positive???
        output = tf.compat.v1.add(tf.compat.v1.matmul(first_out, w2, a_is_sparse = True, b_is_sparse = True), b2)
    else:
        output = first_out
    
    yhat = tf.compat.v1.nn.softmax(output)
    
    if loss=='mse':
        loss = tf.compat.v1.losses.mean_squared_error(y,yhat)
    elif loss == 'cross':
        loss = tf.compat.v1.losses.softmax_cross_entropy(y,yhat)
    
    regularization = 0.0
    if l1 or l1_path:
        regularization = tf.compat.v1.add(regularization, tf.compat.v1.scalar_mul(reg_in, tf.compat.v1.reduce_sum(w1)))
    if l2 or l2_path:
        regularization = tf.compat.v1.add(regularization, tf.compat.v1.scalar_mul(reg_in, tf.compat.v1.reduce_sum(tf.compat.v1.square(w1))))

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
    
    if l1_path or l2_path:
        return {'x':x, 'y':y, 'reg':reg_in,
            'train_step':train_step, 'prediction':prediction, 
            'loss':loss, 'accuracy':accuracy, 'rules':rules, 
            'w1':w1, 'b1':b1, 'w2':w2, 'b2':b2}
    else:
        return {'x':x, 'y':y,'reg':reg_in,'reg_val':l1 if l1 else l2,
            'train_step':train_step, 'prediction':prediction, 
            'loss':loss, 'accuracy':accuracy, 'rules':rules, 
            'w1':w1, 'b1':b1, 'w2':w2, 'b2':b2}

def get_weights(sess, tensors, deep):
    weights = {}
    weights['w1'] = sess.run(tensors['w1'])
    weights['b1'] = sess.run(tensors['b1'])
    if deep:
        weights['w2'] = sess.run(tensors['w2'])
        weights['b2'] = sess.run(tensors['b2'])
    return weights

def train_model_instance(tensors, reg_val, epochs, batch_size, x_data, y_data, deep, patience, delta, patience_metric,
                         x_test, y_test, x_val, y_val, init_vars, verbose, sess):
    indices = list(range(len(y_data)))
    history = {'loss':[]}
    best, best_weights, best_epoch, wait = 1000000, None, 0, 0
    last_acc, last_num_rules = 0, 0
    batch_size = min(batch_size, len(indices))
    if init_vars:
        sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(epochs):
        epoch_loss, epoch_acc = 0, 0
        num_batches = (len(indices) - len(indices) % batch_size) / batch_size
        for start_index in range(0,len(indices) - len(indices) % batch_size, batch_size):
            batch_x, batch_y = get_x_batch(x_data, start_index, batch_size), y_data[start_index:(start_index+batch_size)]
            feed_dict = {tensors['x']:batch_x, tensors['y']:batch_y,tensors['reg']:reg_val}
            _, loss,acc,num_rules = sess.run([tensors['train_step'], tensors['loss'], 
                                              tensors['accuracy'], tensors['rules']], feed_dict)
            epoch_loss += loss
            epoch_acc += acc
            last_acc, last_num_rules = acc, num_rules
        epoch_loss  /= num_batches
        epoch_acc /= num_batches
        history['loss'].append(epoch_loss)
        if patience is not None:
            if history[patience_metric][-1]+delta <= best:
                best = history[patience_metric][-1]
                best_weights = get_weights(sess,tensors,deep)
                best_epoch = epoch
            else:
              wait += 1
              if wait >= patience:
                  if verbose:
                      print(f'Early stopping training of the model at epoch {epoch} with {patience_metric} of {best:.2f}.')
                      print(history,best_weights['w1'].mean())
                  return {'weights':best_weights,'history':{x:history[x] for x in history}}
    if verbose:
        print('Warning - early stopping not triggered. Model may not have converged.')
    return {'weights':best_weights,'history':{x:history[x][:best_epoch] for x in history}}

def train(tensors, epochs, batch_size, x_data, y_data, deep, patience = 3, delta = 0, patience_metric = 'loss', prod = False, 
          x_test = None, y_test = None, x_val = None, y_val = None, l1_path = False, l2_path = False, warm_restart = True,
          verbose = False):
    reg_path = l1_path or l2_path
    with tf.compat.v1.Session() as sess:
        if reg_path:
            reg_vals = [1e-7,1e-6,1e-5,1e-4,1e-3]
            weights =  [None,None,None,None,None]
            history =  [None,None,None,None,None]
            for i in range(len(reg_vals)):
                if verbose:
                    print(f'Running the model with regularization {reg_vals[i]}.')
                init_vars = i==0 or not warm_restart
                results = train_model_instance(tensors, reg_vals[i], epochs//len(reg_vals), batch_size, x_data, y_data,
                                               deep, patience, delta, patience_metric, x_test, y_test, x_val, y_val,
                                               init_vars, verbose, sess)
                weights[i] = results['weights']
                history[i] = results['history']
            best_idx = np.argmin([h[patience_metric][-1] for h in history])
            best_weights = weights[best_idx]
            if verbose:
                print(f'Best performance for model with regularization of {reg_vals[best_idx]}.')
            return {'weights':best_weights,'history':history}
        else:
            reg_val = tensors['reg_val']
            return train_model_instance(tensors, reg_val, epochs, batch_size, x_data, y_data, deep, patience, delta,
                                        patience_metric, x_test, y_test, x_val, y_val, True, verbose, sess)
        

                
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
            weights = get_weights(sess,tensors,deep)
            
            return {'weights':weights, 'history':history}

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
