get_batches <- function(total_size,batch_size) {
  total_indices <- 1:total_size
  shuffled_indices <- sample(total_indices)
  split_indices <- split(shuffled_indices,rep(1:(length(shuffled_indices)/batch_size),length(shuffled_indices)/batch_size))
  return(split_indices)
  
}

get_batch <- function(sparse_matrix, batch_indices=NULL) {
  if(is.null(batch_indices)) {
    res <- as.matrix(1*sparse_matrix)
    return(res)
  }
  sparse_result <- sparse_matrix[batch_indices,]
  res <- as.matrix(1*sparse_result)
  return(res)
}

