#Fill in any missing parameters with their default values
fill_params <- function(params) {
    if(is.null(params)) params <- list()
    if(!'support' %in% names(params)) params$support <- 0.3
    if(!'confidence' %in% names(params)) params$confidence <- 0.5
    if(!'weight_initialization' %in% names(params)) params$weight_initialization <- 'confidence'
    if(!'loss' %in% names(params)) params$loss <- "cross"
    if(!'optimizer' %in% names(params)) params$optimizer <- "sgd"
    if(!'regularization' %in% names(params)) params$regularization <- "none"
    if(!'epoch' %in% names(params)) params$epoch <- 5
    if(!'batch_size' %in% names(params)) params$batch_size <- 16
    if(!'learning_rate' %in% names(params)) params$learning_rate <- 0.001
    return(params)
}


#Train a CWAR classifier and return it represented as a CBA object with an added history field
CWAR <- function(formula, data, params = NULL, verbosity=0) {
  params <- fill_params(params)
  #step 1: mine rules
  apriori_params = list(supp=params$support, conf=params$confidence)
  rules <- mineCARs(formula, data, balanceSupport=T, parameter=apriori_params,control=list(verbose=verbosity>=2))
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
    
    y_ <- tf$placeholder(tf$float64, shape(NULL, num_classes),name='y-tensor')
    T_ <- tf$placeholder(tf$float64, shape(NULL,num_rules),name='T-tensor')
    W <- tf$Variable(class_rules,name='W-tensor')
    if('patience' %in% names(params)) {
      saver <<- tf$train$Saver(max_to_keep=params$patience+2L)
    } else {
      saver <<- tf$train$Saver()
    }
    
    W <- tf$nn$relu(W)
    first_out <- tf$matmul(T_,W,a_is_sparse = F, b_is_sparse = F,name='yhat-tensor')
    yhat <- tf$nn$softmax(first_out,name='softmax-tensor')
    
    if(params$loss=='mse') {
      loss <- tf$losses$mean_squared_error(y_,yhat)
    } else if(params$loss=='cross') {
      loss <- tf$losses$softmax_cross_entropy(y_,yhat)
    } else {
      stop('Error - please specify a valid loss function, params$loss')
    }
    
    if(params$regularization=='l1') {
      if(!'regularization_weights' %in% names(params) || !'l1' %in% names(params$regularization_weights)) {
        stop('Error - please specify params$regularization_weights$l1')
      }
      regularizer <- tf$scalar_mul(params$regularization_weights$l1,tf$reduce_sum(W))
      loss <- tf$add(loss,regularizer)
    } else if(params$regularization=='l2') {
      if(!'regularization_weights' %in% names(params) || !'l2' %in% names(params$regularization_weights)) {
        stop('Error - please specify params$regularization_weights$l2')
      }
      regularizer <- tf$scalar_mul(params$regularization_weights$l2,tf$reduce_sum(tf$square(W)))
      loss <- tf$add(loss,regularizer)
    } else if(params$regularization=='elastic') {
      if(!'regularization_weights' %in% names(params) || !'l2' %in% names(params$regularization_weights) ||
         !'l1' %in% names(params$regularization_weights)) {
        stop('Error - please specify params$regularization_weights$l2 and params$regularization_weights$l1')
      }
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
    accuracy <- tf$reduce_mean(tf$cast(correct_prediction,tf$float64),name='accuracy-tensor')
    
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
               feed_dict = dict(T_=current_batch,y_=get_batch(trans_labels,batch)))
        epoch_loss[[i]] <- epoch_loss[[i]] + results[[2]]
        epoch_accs[[i]] <- epoch_accs[[i]] + results[[3]]
        
        rm(current_batch)
      }
      epoch_loss[[i]] <- epoch_loss[[i]]/length(batches)
      epoch_accs[[i]] <- epoch_accs[[i]]/length(batches)
      epoch_rules[[i]] <- sess$run(tf$count_nonzero(tf$nn$relu(W)))
    }
    
    
    num_rules <- epoch_rules[[params$epoch]]
    weights <- sess$run(W)
    rule_weights <- rowSums(weights)
    
    model <- CBA_ruleset(formula, rules[rule_weights>0], method = 'majority',
                                    weights = rule_weights[rule_weights>0], description = 'CWAR rule set')
    model$history <- list(loss=epoch_loss,accuracy=epoch_accs,rules=epoch_rules)
  })
  #step 6: return CBA object
  model 
}
