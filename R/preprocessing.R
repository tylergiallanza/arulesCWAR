find_rules_per_class <- function(formula,data,rules,method='confidence') {
  class_names <- .parseformula(formula,data)$class_names
  uniform <- as(rhs(rules)[,class_names],'matrix')*1
  if(method=='uniform') {
    return(uniform)
  }else if(method=='confidence') {
    return(uniform*quality(rules)$confidence)
  } else if(method=='random') {
    return(uniform*runif(dim(uniform)[1]))
  }
}

find_rules_per_transaction <- function(rules,transactions) {
  sa <- is.subset(lhs(rules),transactions)
  t(as.matrix(sa))
}

generate_labels <- function(formula,data) {
  class_names <- .parseformula(formula, data)$class_names
  class_columns <- data[,class_names]
  as(class_columns,'matrix')*1 #FIX
}

gen_starting_weights <- function(weight_mask) {
  return(weight_mask)
}