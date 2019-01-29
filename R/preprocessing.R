find_rules_per_class <- function(formula,data,rules,method='confidence') {
  class_names <- .parseformula(formula,data)$class_names
  uniform <- as(rhs(rules)[,class_names],'matrix')*0+1
  if(method=='uniform') {
    return(uniform)
  }else if(method=='confidence') {
    return(uniform*quality(rules)$confidence)
  } else if(method=='random') {
    return(uniform*runif(dim(uniform)[1]))
  }
}

initialize_hidden_weights <- function(num_previous,num_hidden) {
  tf$truncated_normal(c(num_previous,num_hidden), stddev=0.1, dtype = tf$float64)
}

find_rules_per_transaction <- function(rules,transactions) {
  sa <- is.subset(lhs(rules),transactions)
  #t(as.matrix(sa))
  Matrix::t(sa)
}

generate_labels <- function(formula,data) {
  class_names <- .parseformula(formula, data)$class_names
  class_columns <- data[,class_names]
  as(class_columns,'matrix')*1 #FIX
}

gen_starting_weights <- function(weight_mask) {
  return(weight_mask)
}

transactions_to_labels <- function(formula,data) {
  labels <- generate_labels(formula,data)
  l <- unlist(lapply(strsplit(.parseformula(formula,data)$class_names,'='),function(x) x[2]))
  apply(labels, 1, function(x) factor(strsplit(paste(colnames(labels)[which(x == 1)], collapse = ", "),'=')[[1]][2], levels=l )) 
}


tyPredict <- function(object, newdata, ...){
  
  method <- object$method
  if(is.null(method)) method <- "majority"
  
  # weightedmean is just an alias for majority with weights.
  methods <- c("first", "majority", "weighted") # , "weightedmean")
  m <- pmatch(method, methods)
  if(is.na(m)) stop("Unknown method")
  method <- methods[m]
  
  if(!is.null(object$discretization))
    newdata <- discretizeDF(newdata, lapply(object$discretization,
                                            FUN = function(x) list(method="fixed", breaks=x)))
  
  # If new data is not already transactions:
  # Convert new data into transactions and use recode to make sure
  # the new data corresponds to the model data
  newdata <- as(newdata, "transactions")
  newdata <- recode(newdata, match = lhs(object$rules))
  
  # Matrix of which rules match which transactions (sparse is only better for more
  # than 150000 entries)
  rulesMatchLHS <- is.subset(lhs(object$rules), newdata, sparse = F)#,
                             #sparse = (length(newdata) * length(rules(object)) > 150000))
  dimnames(rulesMatchLHS) <- list(NULL, NULL)
  
  class_levels <- sapply(strsplit(object$class, '='), '[',2)
  print(class_levels)
  classifier.results <- unlist(as(rhs(object$rules), "list"))
  classifier.results <- sapply(strsplit(classifier.results, '='), '[', 2)
  classifier.results <- factor(classifier.results, levels = class_levels)
  
  # Default class
  default <- strsplit(object$default, '=')[[1]][2]
  defaultLevel <- which(class_levels == default)
  
  print(defaultLevel)
  
  
  # For each transaction, if it is matched by any rule, classify it using
  # the majority, weighted majority or the highest-precidence
  # rule in the classifier
  
  
  if(method == "majority" | method == "weighted") { # | method == "weightedmean") {
    
    weights <- object$weights
    
    # use a quality measure
    if(is.character(weights))
      weights <- quality(object$rules)[[weights, exact = FALSE]]
    
    # replicate single value (same as unweighted)
    if(length(weights) == 1) weights <- rep(weights, length(object$rules))
    
    # unweighted (use weights of 1)
    if(is.null(weights)) weights <- rep(1, length(object$rules))
    
    # check
    if(length(weights) != length(object$rules))
      stop("length of weights does not match number of rules")
    
    scores <- sapply(1:length(levels(classifier.results)), function(i) {
      classRuleWeights <- weights
      classRuleWeights[as.integer(classifier.results) != i] <- 0
      classRuleWeights %*% rulesMatchLHS
    })
    
    # make sure default wins for ties
    #print(c(class(scores),class(weights),class(rulesMatchLHS)))
    scores[,defaultLevel] <- scores[,defaultLevel] + .Machine$double.eps
    
    output <- factor(apply(scores, MARGIN = 1, which.max),
                     levels = 1:length(levels(classifier.results)),
                     labels = levels(classifier.results))
    
    return(output)
    
    
  }else { ### method = first
    w <- apply(rulesMatchLHS, MARGIN = 2, FUN = function(x) which(x)[1])
    output <- classifier.results[w]
    output[is.na(w)] <- default
  }
  
  
  # preserve the levels of original data for data.frames
  output <- factor(output, levels = class_levels)
  
  return(output)
  
}
