### find X
get_x_data <- function(rules, transactions) t(is.subset(lhs(rules), transactions, sparse = TRUE))

### Y as a dense matrix
get_y_data <- function(formula, transactions) {
  class_ids <- .parseformula(formula, transactions)$class_ids
  class_columns <- as(transactions[,class_ids], "matrix")
  storage.mode(class_columns) <- "numeric"
  class_columns
}

CWAR <- function(formula, data, 
  training_parameter = NULL, mining_parameter = NULL, mining_control = NULL, 
  balanceSupport = TRUE, disc.method = "mdlp", 
  verbose = FALSE, ...) {
  
  source_python(system.file("python", "CWAR_tf.py", package = "arulesCWAR"))
  
  training_params <- .get_parameters(training_parameter, list(
    loss = "cross",
    l1 = 0,
    l2 = 0,
    optimizer = "sgd",
    opt_params = 0.1, ### for sgd
    batch_size = 16,
    epochs = 100,
    learning_rate = 0.001,
    groups = 5,
    patience = 3,
    patience_metric = 'loss',
    patience_delta = 0,
    l1_path = TRUE,
    l2_path = FALSE
  ))
  
  if(verbose) cat("CWAR\n")
  if(verbose) cat("Training parameters:\n")
  print(sapply(training_params, as.character))
  
  formula <- as.formula(formula)
  trans <- prepareTransactions(formula, data, disc.method)
  
  parsed_formula <- .parseformula(formula, trans)
  class <- sapply(strsplit(parsed_formula$class_name, '='), '[', 2)
  
  
  #step 1: mine rules
  if(verbose) cat("1. Mining CARs...\n")
  t1 <- proc.time()
  if(is.null(mining_control)) mining_control <- list(verbose = verbose)
  
  rules <- mineCARs(formula, trans, 
    parameter = mining_parameter, control = mining_control, 
    balanceSupport = balanceSupport, verbose = verbose, ...)
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
    training_params$l1, training_params$l2,
    l1_path = training_params$l1_path,
    l2_path = training_params$l2_path)
  t4 <- proc.time()
  if(verbose) cat("[", t4[3]-t3[3], "s]", "\n", sep ="")
  
  #step 4: run training
  if(verbose) cat("4. Running optimization. ")
  train_data <- train(arch, as.integer(training_params$epochs), as.integer(training_params$batch_size), 
    x_data, y_data, 
    deep = training_params$groups, 
    patience = training_params$patience,
    patience_metric = training_params$patience_metric,
    delta = training_params$patience_delta, 
    l1_path = training_params$l1_path,
    l2_path = training_params$l2_path,
    verbose = verbose)
  weights <- train_data$weights
  history <- train_data$history
  t5 <- proc.time()
  if(verbose) cat("[", t5[3]-t4[3], "s]", "\n", sep ="")
  if(verbose && training_params$groups > 0) cat("  Used groups: ", sum(colSums(weights$w1)>0),
    "/",  ncol(weights$w1), "\n", sep = "")
   
  #step 5: construct model
  softmax <- exp(weights$w1)/rowSums(exp(weights$w1))
  quality(rules)$weight <- apply(softmax, MARGIN = 1, max)
  
  rules_used = sort(rules[rowSums(weights$w1) != 0], by = "weight")
  if(verbose) cat("  Used CARs: ", length(rules_used), "\n", sep = "")
   
  ### MFH: We should use the CBA_ruleset constructor here and put most info into a list in model
  
  structure(list(
    rules = rules_used,
    default = NA,
    discretization = attr(trans, "disc_info"),
    formula = formula,
    method = "logit (CWAR)",
    weights = weights, ### MFH: reduce weights to non-zero weights (for rules_used)
    all_rules = rules, ### MFH: this should go into model
    model = list( 
      parameter = c(list(...), 
        mining_parameter, 
        list(training_params = training_params)
      ),
      history = history
    ),
    description = paste0("CWAR algorithm (unpublished)")
  ), class = c("CWAR", "CBA"))
  
}

# we do the prediction in R
predict.CWAR <- function(object, newdata, type = "class", ...) {
  
  newdata <- prepareTransactions(object$formula, newdata, 
    disc.method = object$discretization,
    match = object$rules)
  
  # find class label for each rule
  RHSclass <- response(object$formula, object$rules)
  
  # MFH: use rules instead of all_rules, but the weights have to be reduced first.
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
  z <- zapsmall(z)
  
  colnames(z) <- levels(RHSclass)
   
  if(type == "class") {
    z <- apply(z, MARGIN = 1, which.max)
    z <- factor(z, 
      levels = 1:length(levels(RHSclass)),
      labels = levels(RHSclass)
    )
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
