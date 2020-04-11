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
CWAR <- function(formula, data, support = 0.1, confidence = 0.5, 
  disc.method = "mdlp", balanceSupport = TRUE, 
  training_parameter = NULL, mining_parameter = NULL, mining_control = NULL, 
  verbose = FALSE) {
  
  source_python(system.file("python", "CWAR_tf.py", package = "arulesCWAR"))
  
  training_params <- .get_parameters(training_parameter, list(
    loss = "cross",
    l1 = 0,
    l2 = 0,
    optimizer = "sgd",
    opt_params = 0.1, ### for sgd
    batch_size = 16,
    epochs = 5,
    learning_rate = 0.001,
    groups = 5
  ))
  if(verbose) cat("CWAR\n")
  if(verbose) cat("Training parameters:\n")
  print(sapply(training_params, as.character))
  
  formula <- as.formula(formula)
  
  # prepare data
  disc_info <- NULL
  if(is(data, "data.frame")){
    data <- discretizeDF.supervised(formula, data, method = disc.method)
    disc_info <- lapply(data, attr, "discretized:breaks")
  }
  
  # convert to transactions for rule mining
  trans <- as(data, "transactions")
  
  parsed_formula <- .parseformula(formula, trans)
  class <- sapply(strsplit(parsed_formula$class_name, '='), '[', 2)
  
  
  #step 1: mine rules
  if(verbose) cat("1. Mining CARs")
  t1 <- proc.time()
  if(is.null(mining_control)) mining_control <- list(verbose = FALSE)
  
  rules <- mineCARs(formula, trans, balanceSupport = balanceSupport, 
    support = support, confidence = confidence,
    parameter = mining_parameter,
    control = mining_control)
  t2 <- proc.time()
 
  if(verbose) cat(": ", length(rules), " [", t2[3]-t1[3],"s]","\n", sep ="")
  if(verbose) cat("  Rules per class:\n")
  if(verbose) print(itemFrequency(rhs(rules)[,parsed_formula$class_ids], type = "absolute"))
   
  #step 2: get training data
  if(verbose) cat("2. Caculating rule coverage. ")
  x_data <- get_x_data(rules, trans)
  x_data <- r_to_py_scipy_sparse(x_data@i, x_data@p)
  y_data <- r_to_py(get_y_data(formula, trans))
  t3 <- proc.time()
  if(verbose) cat("[", t3[3]-t2[3], "s]", "\n", sep ="")
   
  #step 3: build architecture
  if(verbose) cat("3. Building tensorflow architecture. ")
  arch <- build_arch(length(rules), y_data$shape[1], 
    as.integer(training_params$groups), training_params$loss, training_params$optimizer, 
    training_params$opt_params, 
    training_params$l1, training_params$l2)
  t4 <- proc.time()
  if(verbose) cat("[", t4[3]-t3[3], "s]", "\n", sep ="")
  
  #step 4: run training
  if(verbose) cat("4. Running optimization. ")
  weights <- train(arch, as.integer(training_params$epochs), as.integer(training_params$batch_size), 
    x_data, y_data, 
    deep = training_params$groups)
  t5 <- proc.time()
  if(verbose) cat("[", t5[3]-t4[3], "s]", "\n", sep ="")
  if(verbose && training_params$groups > 0) cat("  Used groups: ", sum(colSums(weights$w1)>0),
    "/",  ncol(weights$w1), "\n", sep = "")
   
  #step 5: construct model
  softmax <- exp(weights$w1)/rowSums(exp(weights$w1))
  quality(rules)$weight <- apply(softmax, MARGIN = 1, max)
  
  rules_used = sort(rules[rowSums(weights$w1) != 0], by = "weight")
  if(verbose) cat("  Used CARs: ", length(rules_used), "\n", sep = "")
   
  structure(list(
    rules = rules_used,
    class = class,
    discretization = disc_info,
    formula = formula,
    method = "logit (CWAR)",
    weights = weights,
    all_rules = rules,
    parameters = list(
      support = support,
      confidence = confidence,
      training_params = training_params
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
