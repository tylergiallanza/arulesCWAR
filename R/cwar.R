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

### find X
get_x_data <- function(rules, transactions) t(is.subset(lhs(rules), transactions, sparse = TRUE))

### Y as a dense matrix
get_y_data <- function(formula, transactions) {
  class_ids <- .parseformula(formula, transactions)$class_ids
  class_columns <- as(transactions[,class_ids], "matrix")
  storage.mode(class_columns) <- "numeric"
  class_columns
}

# TODO: make interface look like CBA
CWAR <- function(formula, data, support = 0.1, confidence = 0.5, disc.method = "mdlp", balancedSupport = TRUE, 
  training_params, opt_params, verbose = FALSE) {
  
  source_python(system.file("python", "CWAR_tf.py", package = "arulesCWAR"))
  
  formula <- as.formula(formula)
  
  # prepare data
  disc_info <- NULL
  if(is(data, "data.frame")){
    data <- discretizeDF.supervised(formula, data, method = disc.method)
    disc_info <- lapply(data, attr, "discretized:breaks")
  }
  
  # convert to transactions for rule mining
  trans <- as(data, "transactions")
  
  training_params <- fill_training_params(training_params)
  
  #step 1: mine rules
  rules <- mineCARs(formula, trans, balanceSupport = balancedSupport, support = support, confidence = confidence,
    control = list(verbose = verbose))
  
  #step 2: get training data
  x_data <- get_x_data(rules, trans)
  x_data <- r_to_py_scipy_sparse(x_data@i, x_data@p)
  y_data <- r_to_py(get_y_data(formula, trans))
  
  #step 3: build architecture
  arch <- build_arch(length(rules), y_data$shape[1], 
    training_params$deep, training_params$loss, training_params$optimizer, 
    opt_params, 
    training_params$l1, training_params$l2)
  
  #step 4: run training
  weights <- train(arch, training_params$epochs, training_params$batch_size, 
    x_data, y_data, 
    deep = training_params$deep)
  
  #step 5: construct model
  parsed_formula <- .parseformula(formula, trans)
  class <- sapply(strsplit(parsed_formula$class_name, '='), '[', 2)
 
  softmax <- exp(weights$w1)/rowSums(exp(weights$w1))
  quality(rules)$weight <- apply(softmax, MARGIN = 1, max)
   
  structure(list(
    rules = rules[rowSums(weights$w1) != 0],
    class = class,
    discretization = disc_info,
    formula = formula,
    method = "logit (CWAR)",
    weights = weights,
    all_rules = rules,
    parameters = list(
      support = support,
      confidence = confidence,
      training_params = training_params,
      opt_params = opt_params
    ),
    description = paste0("CWAR algorithm with support=", support,
      " and confidence=", confidence)
  ), class = c("CWAR", "CBA"))
  
}

# we do the prediction in R
# TODO: we can save time by matching agains rules instead of all_rules
predict.CWAR <- function(object, newdata, type = "class", ...) {
  
  if(!is.null(object$discretization)) {
    newdata <- discretizeDF(newdata, lapply(object$discretization,
      FUN = function(x) list(method="fixed", breaks=x)))
    newdata <- as(newdata, "transactions")
  } else {
    if(!is(newdata, "transactions"))
      stop("Classifier does not contain discretization information. New data needs to be in the form of transactions. Check ? discretizeDF.")
  }
  
  # If new data is not already transactions:
  # Convert new data into transactions and use recode to make sure
  # the new data corresponds to the model data
  newdata <- recode(newdata, match = lhs(object$rules))
  
  x_data <- get_x_data(object$all_rules, newdata)
  dimnames(x_data) <- list(NULL, NULL)
  
  # layer 1
  z <- x_data %*% object$weights$w1
  z <- sweep(z, MARGIN = 2, STATS = object$weights$b1, FUN = "+") 
  
  # layer 2
  if(!is.null(object$weights$w2)) {
    z <- z %*% object$weights$w2
    z <- sweep(z, MARGIN = 2, STATS = object$weights$b2, FUN = "+") 
  }
  
  
  # softmax
  z <- exp(z)
  z <- z / rowSums(z)
  
  colnames(z) <- object$class
  
  if(type == "class") {
    z <- apply(z, MARGIN = 1, which.max)
    z <- factor(z, levels = 1:length(object$class), labels = object$class)
  }
  
  z
}

# use tensorflow
# predict.CWAR <- function(object, newdata, ...) {
#   source_python(system.file("python", "CWAR_tf.py", package = "arulesCWAR"))
#   
#  x_data <- get_x_data(object$rules,newdata)
#  x_data <- r_to_py_scipy_sparse(x_data@i, x_data@p)
#
#   tensors <- build_arch_for_object(length(object$rules), 
#     object$weights$w1, object$weights$w2, object$weights$b1, object$weights$b2)
#   yhat <- eval_tensors(tensors,'yhat', x_data)
#   
#   factor(object$class[yhat+1], levels = object$class)
# }
