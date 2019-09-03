get_batches <- function(total_size,batch_size, shuffle = T) {
  total_indices <- 1:total_size
  if(shuffle) {
    shuffled_indices <- sample(total_indices)
    split_indices <- split(shuffled_indices,rep(1:(length(shuffled_indices)/batch_size),length(shuffled_indices)/batch_size))
    return(split_indices)
  } else {
    #split_indices <- split(total_indices,rep(1:(length(total_indices)/batch_size),length(total_indices)/batch_size))
    split_indices <- split(total_indices, ceiling(seq_along(total_indices)/batch_size))
    return(split_indices)
  }
  
}

get_batch <- function(sparse_matrix, batch_indices=NULL, sparse=F) {
  if(is.null(batch_indices)) {
    if (sparse){
      return(sparse_matrix)
    } else{
      res <- as.matrix(1*sparse_matrix)
      return(res)
    }
  }
  if(sparse){
    py_idx <- r_to_py(batch_indices-1)
    return(sparse_matrix[py_idx])
  } else {
    sparse_result <- sparse_matrix[batch_indices,]
    res <- as.matrix(1*sparse_result)
    return(res)
  }
}
