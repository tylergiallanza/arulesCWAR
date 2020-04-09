#Fill in any missing parameters with their default values
fill_mining_params <- function(mining_params) {
    if(is.null(mining_params)) mining_params <- list()
    if(!'support' %in% names(mining_params)) mining_params$support <- 0.3
    if(!'confidence' %in% names(mining_params)) mining_params$confidence <- 0.5
    return(mining_params)
}
    
fill_training_params <- function(params) {
    if(is.null(params)) params <- list()
    if(!'deep' %in% names(params)) params$deep <- 5L
    if(!'loss' %in% names(params)) params$loss <- "cross"
    if(!'optimizer' %in% names(params)) params$optimizer <- "sgd"
    if(!'l1' %in% names(params)) params$l1 <- 0
    if(!'l2' %in% names(params)) params$l2 <- 0
    if(!'epochs' %in% names(params)) params$epochs <- 5L
    if(!'batch_size' %in% names(params)) params$batch_size <- 16L
    if(!'learning_rate' %in% names(params)) params$learning_rate <- 0.001
    return(params)
}

get_x_data <- function(rules, transactions) {
  source_python('tf.py')
  rpt <- Matrix::t(is.subset(lhs(rules),transactions)) #CHECK THIS
  py_rpt <- process_sparse(rpt@i,rpt@p)
  return(py_rpt)
}


get_y_data <- function(formula, data) {
  class_names <- .parseformula(formula, data)$class_names
  class_columns <- data[,class_names]
  return(r_to_py(as(class_columns,'matrix')*1))
}


CWAR <- function(formula, data, mining_params, training_params, opt_params) {
  source_python('tf.py')
  mining_params <- fill_mining_params(mining_params)
  training_params <- fill_training_params(training_params)
  #step 1: mine rules
  rules <- mineCARs(formula, data, balanceSupport=T,parameter=mining_params,control=list(verbose=F))
  #step 2: get training data
  x_data <- get_x_data(rules, data)
  y_data <- get_y_data(formula, data)
  #step 3: build architecture
  arch <- build_arch(length(rules),y_data$shape[1],training_params$deep, training_params$loss, training_params$optimizer, opt_params, training_params$l1, training_params$l2)
  #step 4: run training
  weights <- train(arch, training_params$epochs, training_params$batch_size, x_data, y_data, deep=training_params$deep)
  #step 5: save model
  model <- list()
  model$original_rules <- rules
  model$mining_params <- mining_params
  model$training_params <- training_params
  model$opt_params <- opt_params
  model$formula <- formula
  model$weights <- weights
  model$class_names <- unlist(lapply(strsplit(.parseformula(formula,data)$class_names,'='), function(x) x[2]))
  model$num_rules <- sum(rowSums(model$weights$w1)!=0)
  class(model) <- 'CWAR'
  return(model)
}

predict.CWAR <- function(object, newdata, ...) {
  source_python('tf.py')
  tensors <- build_arch_for_object(length(object$original_rules),object$weights$w1, object$weights$w2, object$weights$b1, object$weights$b2)
  yhat <- eval_tensors(tensors,'yhat',get_x_data(object$original_rules,newdata))
  return(factor(object$class_names[yhat+1],levels=object$class_names))
}
