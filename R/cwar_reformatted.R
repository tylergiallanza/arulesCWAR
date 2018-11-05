
.parseformula <- function(formula, data) {
  formula <- as.formula(formula)
  vars <- all.vars(formula)
  
  ### class
  # for transactions, class can match multiple items!
  class <- vars[1]
  
  if(is(data, "transactions")) {
    class_ids <- which(grepl(paste0("^", class), colnames(data)))
  } else {
    class_ids <- pmatch(class, colnames(data))
  }
  if(is.na(class_ids) || length(class_ids) == 0)
    stop("Cannot identify column specified as class in the formula.")
  class_names <- colnames(data)[class_ids]
  
  if(!is(data, "transactions") && !is.factor(data[[class_ids]]))
    stop("class variable needs to be a factor!")
  
  ### predictors
  vars <- vars[-1]
  if(is(data, "transactions")) {
    if(length(vars) == 1 && vars == ".") var_ids <- setdiff(seq(ncol(data)), class_ids)
    else var_ids <- which(grepl(paste0("^", vars, collapse = "|"), colnames(data)))
  } else {
    if(length(vars) == 1 && vars == ".") var_ids <- setdiff(which(sapply(data, is.numeric)), class_ids)
    else var_ids <- pmatch(vars, colnames(data))
  }
  
  if(any(is.na(var_ids))) stop(paste("Cannot identify term", vars[is.na(var_ids)], "in data! "))
  var_names <- colnames(data)[var_ids]
  
  
  list(class_ids = class_ids, class_names = class_names,
       var_ids = var_ids, var_names = var_names,
       formula = formula)
}

find_rules_per_class <- function(formula,data,rules,method='confidence') {
  class_names <- .parseformula(formula,data)$class_names
  uniform <- as(rhs(rules)[,class_names],'matrix')*1
  if(method=='uniform') {
    return(uniform)
  }else if(method=='confidence') {
    return(uniform*quality(rules)$confidence)
  }
}

find_rules_per_transaction <- function(rules,transactions) {
  sa <- is.subset(lhs(rules),transactions)
  t(as.matrix(sa)) #FIX THIS SO IT STAYS SPARSE
}

generate_labels <- function(formula,data) {
  class_names <- .parseformula(formula, data)$class_names
  class_columns <- data[,class_names]
  as(class_columns,'matrix')*1 #FIX
}

gen_starting_weights <- function(weight_mask) {
  return(weight_mask)
}

get_batches <- function(total_size,batch_size) {
  total_indices <- 1:total_size
  shuffled_indices <- sample(total_indices)
  split_indices <- split(shuffled_indices,rep(1:(length(shuffled_indices)/batch_size),length(shuffled_indices)/batch_size))
  return(split_indices)
  
}

get_batch <- function(sparse_matrix, batch_indices=NULL) {
  #print(paste('sparse matrix size',object.size(sparse_matrix),class(sparse_matrix)))
  if(is.null(batch_indices)) {
    res <- as.matrix(1*sparse_matrix)
    #print(object.size(res))
    return(res)
  }
  sparse_result <- sparse_matrix[batch_indices,]
  res <- as.matrix(1*sparse_result)
  #print(paste('batch size for',length(batch_indices),'items:',object.size(res)))
  return(res)
}

#params: support, confidence, weight_initialization
CWAR <- function(formula, data, params, verbosity=0) {
  #step 1: mine rules
  apriori_params = list(supp=params$support, conf=params$conf)
  rules <- mineCARs2(formula, data, balanceSupport=T, parameter=apriori_params,control=list(verbose=verbosity>=2))
  #step 2: find rules per class
  class_rules <- find_rules_per_class(formula,data,rules,method=params$weight_initialization)
  #step 3: find rules per transaction
  trans_rules <- find_rules_per_transaction(rules,data)
  #step 4: extract y data
  trans_labels <- generate_labels(formula,data)
  #step 5: run CWAR algorithm
  best_epoch <- 1
  tf$reset_default_graph() 
  with(tf$Session() %as% sess,{
    num_rules <- dim(class_rules)[1]
    num_classes <- dim(class_rules)[2]
    
    C_ <- tf$placeholder(tf$float32, shape(num_rules,num_classes),name='C-tensor')
    y_ <- tf$placeholder(tf$float32, shape(NULL, num_classes),name='y-tensor')
    T_ <- tf$placeholder(tf$float32, shape(NULL,num_rules),name='T-tensor')
    W <- tf$Variable(tf$ones(shape(num_rules,num_classes)),name='W-tensor')
    if('patience' %in% names(params)) {
      saver <<- tf$train$Saver(max_to_keep=params$patience+2L)
    } else {
      saver <<- tf$train$Saver()
    }
    W <- tf$multiply(W,C_)
    W <- tf$nn$relu(W)
    yhat <- tf$matmul(T_,W,a_is_sparse = F, b_is_sparse = F,name='yhat-tensor') #TODO: check on sparsity
    
    if(params$loss=='mse') {
      loss <- tf$losses$mean_squared_error(y_,yhat)
    } else if(params$loss=='cross') {
      loss <- tf$losses$softmax_cross_entropy(y_,yhat)
    }
    #TODO: add checks for params
    
    if(params$regularization=='l1') {
      regularizer <- tf$scalar_mul(params$regularization_weights$l1,tf$reduce_sum(W))
      loss <- tf$add(loss,regularizer)
    } else if(params$regularization=='l2') {
      regularizer <- tf$scalar_mul(params$regularization_weights$l2,tf$reduce_sum(tf$square(W)))
      loss <- tf$add(loss,regularizer)
    } else if(params$regularization=='elastic') {
      l1 <- tf$scalar_mul(params$regularization_weights$l1,tf$reduce_sum(W))
      l2 <- tf$scalar_mul(params$regularization_weights$l2,tf$reduce_sum(tf$square(W)))
      loss <- tf$add(tf$add(loss,l1),l2)
    }
    
    if(params$optimizer=='sgd') {
      optimizer <- tf$train$GradientDescentOptimizer(params$learning_rate)
    } else if(params$optimizer=='adam') {
      optimizer <- tf$train$AdamOptimizer(params$adam_params$learning_rate,params$adam_params$beta1,
                                          params$adam_params$beta2,params$adam_params$epsilon)
    } else if(params$optimizer=='adadelta') {
      optimizer <- tf$train$AdadeltaOptimizer(params$adadelta_params$learning_rate,params$adadelta_params$rho,
                                              params$adadelta_params$epsilon)
    } else {
      stop('Error - please specify a valid optimizer!')
    }

    train_step <- optimizer$minimize(loss)
    
    correct_prediction <- tf$equal(tf$argmax(yhat,1L),tf$argmax(y_,1L))
    accuracy <- tf$reduce_mean(tf$cast(correct_prediction,tf$float32),name='accuracy-tensor')
    
    sess$run(tf$global_variables_initializer())
    epoch_loss <- vector('list',params$epoch)
    epoch_accs <- vector('list',params$epoch)
    epoch_rules <- vector('list',params$epoch)
    t <- proc.time()[1]
    for(i in 1:params$epoch) {
      batches <- get_batches(dim(trans_rules)[1],params$batch_size)
      epoch_loss[[i]] <- 0
      epoch_accs[[i]] <- 0
      for(batch in batches) {
        current_batch <- get_batch(trans_rules,batch)
        results <- sess$run(c(train_step,loss,accuracy),
               feed_dict = dict(C_=class_rules,T_=current_batch,y_=get_batch(trans_labels,batch))) #TODO: no feed dict
        epoch_loss[[i]] <- epoch_loss[[i]] + results[[2]]
        epoch_accs[[i]] <- epoch_accs[[i]] + results[[3]]
        
        rm(current_batch)
      }
      epoch_loss[[i]] <- epoch_loss[[i]]/length(batches)
      epoch_accs[[i]] <- epoch_accs[[i]]/length(batches)
      epoch_rules[[i]] <- sess$run(tf$count_nonzero(tf$nn$relu(W)),feed_dict=dict(C_=class_rules))
    }
    model <- list()
    model$num_rules <- epoch_rules[[params$epoch]]
    model$weights <- sess$run(W,feed_dict=dict(C_=class_rules)) #combine this with n_rules above
    model$classifier <- CBA_ruleset(formula, rules[model$weights>0], method = 'majority',
                                    weights = model$weights[model$weights>0], description = 'CWAR rule set')
    model$history <- list(loss=epoch_loss,accuracy=epoch_accs,rules=epoch_rules)
    class(model) <- "CWAR"
  })
  #step 6: return CBA object
  return(model) #FIX
}

predict.CWAR <- function(object, newdata, ...) {
  predict(object$classifier, newdata, ...)
}

#data(Adult)
#cwar_params <- list(support=0.3,confidence=0.5,weight_initialization='confidence',
                    #loss='cross',regularization='l1',optimizer='sgd',epoch=2,batch_size=16,learning_rate=0.001)
#CWAR('income~.',Adult,cwar_params,0)
