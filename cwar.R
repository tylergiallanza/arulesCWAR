library('devtools')
install_github('ianjjohnson/arulesCBA')
library(arulesCBA)
library(randomForest)
library(tensorflow)
library(missForest)

count_na <- function(df) {
  sapply(df, function(y) sum(length(which(is.na(y)))))
}

prepare_poker <- function() {
  poker_data <- read.csv('~/Dropbox/School/CSE5393/sgd/data/poker.data',
                          header=F,sep=',')
  poker_data[,names(poker_data)] <- lapply(poker_data[,names(poker_data)],factor)
  poker_data <- poker_data[1:10000,]
  poker_data <- poker_data[-which(poker_data$V11==7),]
  poker_data <- poker_data[-which(poker_data$V11==8),]
  poker_data <- poker_data[-which(poker_data$V11==9),]
  poker_data
}

prepare_census <- function() {
  census_data <- read.csv('~/Dropbox/School/CSE5393/sgd/data/census-income.data',
                          header=F,sep=',',na.strings=' ?')
  census_data[['V25']] <- NULL
  census_data$V42 <- factor(census_data$V42)
  census_data <- discretizeDF.supervised('V42~.',census_data)
  census_data[1:20000,]
  
}

prepare_anneal <- function() {
  anneal_data <- read.csv('~/Dropbox/School/CSE5393/sgd/data/anneal.data',
                          header=FALSE,sep=',',na.strings='?')
  
  anneal_data[['V1']] <- NULL
  anneal_data$V39 <- factor(anneal_data[,'V39']) #class variable
  anneal_data <- discretizeDF.supervised('V39~.',anneal_data)
  
  
  anneal_data
}

prepare_austral <- function() {
  austral_data <- read.csv(url('http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat'),
                           header=FALSE,sep=' ',na.strings='?')
  austral_data$V15 <- factor(austral_data[,'V15']) #CLASS VARIABLE
  austral_data <- discretizeDF.supervised('V15~.',austral_data)
  austral_data
}

prepare_auto <- function() {
  auto_data <- read.csv('~/Dropbox/School/CSE5393/sgd/data/auto.data',
                        header=FALSE,sep=',',na.strings='?')
  
  temp <- factor(auto_data[['V1']])
  auto_data[['V2']] <- NULL
  auto_data$V26 <- temp #class variable
  auto_data <- discretizeDF.supervised('V26~.',auto_data)
  
  auto_data
}

prepare_bc <- function() {
  bc_data <- read.csv('~/Dropbox/School/CSE5393/sgd/data/bc.data',header=FALSE)
  bc_data$V1 <- NULL
  bc_data[,'V2'] <- factor(bc_data[,'V2'])
  bc_data[,'V3'] <- factor(bc_data[,'V3'])
  bc_data[,'V4'] <- factor(bc_data[,'V4'])
  bc_data[,'V5'] <- factor(bc_data[,'V5'])
  bc_data[,'V6'] <- factor(bc_data[,'V6'])
  bc_data[,'V7'] <- factor(bc_data[,'V7'])
  bc_data[,'V8'] <- factor(bc_data[,'V8'])
  bc_data[,'V9'] <- factor(bc_data[,'V9'])
  bc_data[,'V10'] <- factor(bc_data[,'V10'])
  bc_data[,'V11'] <- factor(bc_data[,'V11'])
  bc_data
}

prepare_crx <- function() {
  crx_data <- read.csv('~/Dropbox/School/CSE5393/sgd/data/crx.data',header=FALSE,
                       na.strings='?')
  crx_data[,'V16'] <- factor(crx_data[,'V16']) #CLASS VARIABLE
  
  crx_data <- discretizeDF.supervised('V16~.',crx_data)
  crx_data
}

prepare_cleve <- function() {
  cleve_data <- read.csv(url('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'),
                         header=F,sep=',')
                   
  change_indices <- c(which(cleve_data$V14!='0'))
  cleve_data[change_indices,'V14'] <- 1
  cleve_data[,'V14'] <- factor(cleve_data[,'V14'])
  cleve_data <- discretizeDF.supervised('V14~.',cleve_data)
  cleve_data
}

prepare_german <- function() {
  german_data <- read.csv('~/Dropbox/School/CSE5393/sgd/data/german.data',header=FALSE,sep=' ')
  german_data$V21 <- factor(german_data[,'V21']) #Class variable
  german_data <- discretizeDF.supervised('V21~.',german_data)
  german_data
}

prepare_glass <- function() {
  glass_data <- read.csv(url('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'),header=FALSE,sep=',')
  glass_data$V1 <- NULL
  glass_data$V11 <- factor(glass_data[,'V11']) #Class variable
  glass_data <- discretizeDF.supervised('V11~.',glass_data)
  glass_data
}

prepare_heart <- function() {
  heart_data <- read.table(url('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat'),
                           sep=' ',header=FALSE)
  heart_data[,'V14'] <- factor(heart_data[,'V14'])
  heart_data <- discretizeDF.supervised('V14~.',heart_data)
  heart_data
}

prepare_hepatic <- function() {
  hepatic_data <- read.csv('~/Dropbox/School/CSE5393/sgd/data/hepatic.data',sep=',',na.strings='?',header=FALSE)
  temp <- factor(hepatic_data[,'V1'])
  hepatic_data$V1 <- hepatic_data[,'V20']
  hepatic_data$V20 <- temp
  hepatic_data <- discretizeDF.supervised('V20~.',hepatic_data)
  hepatic_data
}

prepare_horse <- function() {
  horse_data <- read.table('~/Dropbox/School/CSE5393/sgd/horse.csv',
                           sep=',',header=FALSE,na.strings='?')
  #horse_data <- read.table('D:\\Dropbox\\School\\CSE5393\\sgd\\horse.csv',
  #                         sep=',',header=FALSE,na.strings='?')
  horse_data[,'V3'] <- NULL
  horse_data[,'V24'] <- factor(horse_data[,'V24'])
  horse_data <- discretizeDF.supervised('V24~.',horse_data)
  
  horse_data
}

prepare_iono <- function() {
  iono_data <- read.table('~/Dropbox/School/CSE5393/sgd/data/iono.data',
                           sep=',',header=FALSE)
  iono_data[,'V35'] <- factor(iono_data[,'V35'])
  iono_data <- discretizeDF.supervised('V35~.',iono_data)
  iono_data
}

prepare_iris <- function() {
  iris_data <- read.table('~/Dropbox/CSE5393/sgd/data/iris.data',
                           sep=',',header=FALSE)
  iris_data[,'V5'] <- factor(iris_data[,'V5'])
  iris_data <- discretizeDF.supervised('V5~.',iris_data)
  iris_data[5,'V2'] <- NA
  iris_data
}

prepare_labor <- function() {
  labor_data <- read.csv('~/Dropbox/School/CSE5393/sgd/labor.csv',sep=',',header=FALSE,na.strings='?')
  labor_data$V17 <- factor(labor_data$V17)
  
  labor_data <- discretizeDF.supervised('V17~.',labor_data)
  labor_data
}

prepare_led7 <- function() {
  led7_data <- read.table('~/Dropbox/School/CSE5393/sgd/led7.csv',
                           sep=',',header=FALSE)
  led7_data[,'V1'] <- factor(led7_data[,'V1'])
  led7_data[,'V2'] <- factor(led7_data[,'V2'])
  led7_data[,'V3'] <- factor(led7_data[,'V3'])
  led7_data[,'V4'] <- factor(led7_data[,'V4'])
  led7_data[,'V5'] <- factor(led7_data[,'V5'])
  led7_data[,'V6'] <- factor(led7_data[,'V6'])
  led7_data[,'V7'] <- factor(led7_data[,'V7'])
  led7_data[,'V8'] <- factor(led7_data[,'V8']) #Class variable
  led7_data
}

prepare_lymph <- function() {
  lymph_data <- read.table('~/Dropbox/School/CSE5393/sgd/data/lymph.data',
                           sep=',',header=FALSE)
  temp <- factor(lymph_data[,'V1'])
  lymph_data[,'V1'] <- lymph_data[,'V19']
  lymph_data[,'V19'] <- temp #class variable
  lymph_data <- discretizeDF.supervised('V19~.',lymph_data)
  lymph_data
}

prepare_pima <- function() {
  pima_data <- read.csv(url('https://gist.githubusercontent.com/ktisha/c21e73a1bd1700294ef790c56c8aec1f/raw/819b69b5736821ccee93d05b51de0510bea00294/pima-indians-diabetes.csv'),
                        sep=',',header=FALSE,skip=9)
  pima_data[,'V9'] <- factor(pima_data[,'V9']) #class variable
  pima_data <- discretizeDF.supervised('V9~.',pima_data)
  
  pima_data
}

prepare_sick <- function() {
  sick_data <- read.table('~/Dropbox/School/CSE5393/sgd/data/sick.data',
                           sep=',',header=FALSE,na.strings='?')
  sick_data[,'V30'] <- as.character(levels(sick_data[,'V30']))[sick_data[,'V30']]
  class_labels <- unlist(strsplit(sick_data[,'V30'],"\\."))[1:length(sick_data[,'V30'])*2-1]
  sick_data[,'V30'] <- factor(class_labels)
  sick_data[,'V28'] <- NULL
  sick_data <- discretizeDF.supervised('V30~.',sick_data)
  sick_data
}

prepare_sonar <- function() {
  sonar_data <- read.table(url('http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'),
                           sep=',',header=FALSE)
  sonar_data[,'V61'] <- factor(sonar_data[,'V61']) #class variable
  sonar_data <- discretizeDF.supervised('V61~.',sonar_data)
  
  sonar_data
}

prepare_tic <- function() {
  tic_data <- read.table('~/Dropbox/School/CSE5393/sgd/data/tic.data',
                           sep=',',header=FALSE)
  tic_data[,'V1'] <- factor(tic_data[,'V1'])
  tic_data[,'V2'] <- factor(tic_data[,'V2'])
  tic_data[,'V3'] <- factor(tic_data[,'V3'])
  tic_data[,'V4'] <- factor(tic_data[,'V4'])
  tic_data[,'V5'] <- factor(tic_data[,'V5'])
  tic_data[,'V6'] <- factor(tic_data[,'V6'])
  tic_data[,'V7'] <- factor(tic_data[,'V7'])
  tic_data[,'V8'] <- factor(tic_data[,'V8'])
  tic_data[,'V9'] <- factor(tic_data[,'V9'])
  tic_data[,'V10'] <- factor(tic_data[,'V10'])
  tic_data
}

prepare_wine <- function() {
  wine_data <- read.table('~/Dropbox/School/CSE5393/sgd/data/wine.data',
                           sep=',',header=FALSE)
  temp <- factor(wine_data[,'V1'])
  wine_data[,'V1'] <- wine_data[,'V13']
  wine_data[,'V13'] <- temp
  
  wine_data <- discretizeDF.supervised('V13~.',wine_data)
  wine_data
}

prepare_waveform <- function() {
  waveform_data <- read.table('~/Dropbox/School/CSE5393/sgd/waveform.data',sep=',',header=F,na.strings='?')
  waveform_data$V22 <- factor(waveform_data$V22)
  
  waveform_data <- discretizeDF.supervised('V22~.',waveform_data)
  waveform_data
}

prepare_vehicle <- function() {
  vehicle_data <- read.table('~/Dropbox/School/CSE5393/sgd/vehicle.csv',sep=',',header=F,strip.white=T,
                             na.strings = '?')
  vehicle_data$V19 <- factor(vehicle_data$V19)
  vehicle_data <- discretizeDF.supervised('V19~.',vehicle_data)
  vehicle_data
}

prepare_zoo <- function() {
  zoo_data <- read.table('~/Dropbox/School/CSE5393/sgd/data/zoo.data',
                           sep=',',header=FALSE,na.strings='?')
  zoo_data[,'V1'] <- NULL
  zoo_data[,'V2'] <- factor(zoo_data[,'V2'])
  zoo_data[,'V3'] <- factor(zoo_data[,'V3'])
  zoo_data[,'V4'] <- factor(zoo_data[,'V4'])
  zoo_data[,'V5'] <- factor(zoo_data[,'V5'])
  zoo_data[,'V6'] <- factor(zoo_data[,'V6'])
  zoo_data[,'V7'] <- factor(zoo_data[,'V7'])
  zoo_data[,'V8'] <- factor(zoo_data[,'V8'])
  zoo_data[,'V9'] <- factor(zoo_data[,'V9'])
  zoo_data[,'V10'] <- factor(zoo_data[,'V10'])
  zoo_data[,'V11'] <- factor(zoo_data[,'V11'])
  zoo_data[,'V12'] <- factor(zoo_data[,'V12'])
  zoo_data[,'V13'] <- factor(zoo_data[,'V13'])
  zoo_data[,'V14'] <- factor(zoo_data[,'V14'])
  zoo_data[,'V15'] <- factor(zoo_data[,'V15'])
  zoo_data[,'V16'] <- factor(zoo_data[,'V16'])
  zoo_data[,'V17'] <- factor(zoo_data[,'V17'])
  zoo_data[,'V18'] <- factor(zoo_data[,'V18']) #Class variable
  
  zoo_data
}


generate_folds <- function(data) {
  data_indices <- 1:dim(data)[1]
  shuffled_indices <- sample(data_indices)
  split_indices <- split(shuffled_indices,rep(1:10,length(shuffled_indices)/10))
  return(split_indices)
}

### formula helper
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


mineCARs2 <- function(formula, data, balanceSupport = FALSE,
  parameter = NULL, control = NULL, ...) {

  if(!is(data, "transactions")) stop("data needs to contain an object of class transactions.")
  vars <- .parseformula(formula, data)


  if(!balanceSupport) {
  #Generate CARs with APRIORI
  apriori(data, parameter = parameter,
    appearance = list(rhs=vars$class_names, lhs=vars$var_names),
    control=control, ...)
  }else {

  if(is.null(parameter)) parameter <- new("APparameter")
  else parameter <- as(parameter, "APparameter")

  iSupport <- itemFrequency(data)[vars$class_ids]
  #freq <- itemFrequency(data[,vars$class_ids])

  suppMultiplier <- iSupport/max(iSupport)

  rs <- lapply(names(suppMultiplier), FUN = function(rhs) {

    newParameter <- parameter
    newParameter@support <- newParameter@support * suppMultiplier[[rhs]]

    apriori(data, parameter = newParameter,
      appearance = list(rhs = rhs, lhs=vars$var_names),
      control=control, ...)

  })

  do.call(c, rs)

  }
}

mine_rules_nested <- function(data, data_indices, class_column, sup, conf,maxlength=10) {
  train_data <- as(data[data_indices,],'transactions')
  data_rules <- mineCARs2(paste0(class_column,'~.'),train_data,balanceSupport=T,parameter = 
                            list(support=sup,conf=conf,maxlen=maxlength),control=list(verbose=F))
  data_rules
}


find_rules_per_class <- function(rules,method='confidence') {
  if(method=='confidence') {
    rule_factor <- factor(unlist(as(rhs(rules),'list')))
    num_classes <- length(levels(rule_factor))
    rule_predictions <- as.integer(rule_factor)
    class_rules <- matrix(nrow=length(rules),ncol=num_classes)
    class_rules[,] <- 0
    for(i in 1:num_classes) {
      rule_indices <- which(rule_predictions==i)
      class_rules[rule_indices,i] <- quality(rules)$confidence[rule_indices]
    }
    class_rules
  } else {
    rule_factor <- factor(unlist(as(rhs(rules),'list')))
    num_classes <- length(levels(rule_factor))
    rule_predictions <- as.integer(rule_factor)
    class_rules <- matrix(nrow=length(rules),ncol=num_classes)
    class_rules[,] <- 0
    for(i in 1:num_classes) {
      rule_indices <- which(rule_predictions==i)
      class_rules[rule_indices,i] <- 1
    }
    class_rules
  }
}

find_rules_per_transaction <- function(rules,transactions) {
  t(is.subset(lhs(rules),transactions))
}

generate_labels <- function(itemset,class_column) {
  class_names <- itemset[[class_column]]
  class_numbers <- as.numeric(class_names)
  class_ohe <- matrix(nrow=dim(itemset)[1],ncol=max(class_numbers))
  class_ohe[,] <- 0
  for(i in 1:dim(itemset)[1]) {
    class_ohe[i,class_numbers[i]] <- 1
  }
  class_ohe
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

CWAR <- function(class_rules,train_trans_rules,train_trans_labels,validation_trans_rules,validation_trans_labels,
                 params,verbosity=0,logging=0) {
  model <- list(model=NULL,class_rules=class_rules,validation_accuracy=0,num_rules=0)
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
    epoch_val_accs <- vector('list',params$epoch)
    epoch_rules <- vector('list',params$epoch)
    t <- proc.time()[1]
    for(i in 1:params$epoch) {
      epoch_time <- proc.time()[1]
      batches <- get_batches(dim(train_trans_rules)[1],params$batch_size)
      epoch_loss[[i]] <- 0
      epoch_accs[[i]] <- 0
      for(batch in batches) {
        current_batch <- get_batch(train_trans_rules,batch)
        results <- sess$run(c(train_step,loss,accuracy),
               feed_dict = dict(C_=class_rules,T_=current_batch,y_=get_batch(train_trans_labels,batch))) #TODO: no feed dict
        epoch_loss[[i]] <- epoch_loss[[i]] + results[[2]]
        epoch_accs[[i]] <- epoch_accs[[i]] + results[[3]]
        
        rm(current_batch)
      }
      #print(paste('  tf time',proc.time()[1]-epoch_time))
      epoch_loss[[i]] <- epoch_loss[[i]]/length(batches)
      epoch_accs[[i]] <- epoch_accs[[i]]/length(batches)
      epoch_val_accs[[i]] <- evaluate_model_accuracy(model,validation_trans_rules,validation_trans_labels,batch_size=params$batch_size)
      epoch_rules[[i]] <- sess$run(tf$count_nonzero(tf$nn$relu(W)),feed_dict=dict(C_=class_rules))
      #print(paste('epoch',i,'time',proc.time()[1]-epoch_time))
    }
    print(paste('time',proc.time()[1]-t))
    model$validation_accuracy <- epoch_accs[[params$epoch]]
    model$num_rules <- epoch_rules[[params$epoch]]
    model$weights <- sess$run(W,feed_dict=dict(C_=class_rules)) #combine this with n_rules above
    model$history <- list(loss=epoch_loss,accuracy=epoch_accs,validation_accuracy=epoch_val_accs,rules=epoch_rules)
    return(model)
  })
}

is_better_model <- function(new_acc,new_rules,old_acc,old_rules) {
  if(new_acc+0.002<old_acc) {
    return(F)
  }
  if(new_acc-0.002>old_acc) {
    return(T)
  }
  if(new_rules<old_rules) {
    return(T)
  }
  return(F)
  
}

run_grid_search <- function(class_rules,train_trans_rules,train_trans_labels,validation_trans_rules,validation_trans_labels,
                              parameters,verbosity,logging,history=F,tested_models=list()) {
  best_model <<- list(validation_accuracy=0)
  histories <- vector('list',5)
  param_index <<- 1
  for(loss_function in parameters$loss) {
    for(opt in parameters$optimizer) {
      for(reg in parameters$regularization) {
        for(reg_weights in parameters$regularization_weights) {
          for(ap in parameters$adam_params) {
            for(epoch in parameters$epochs) {
              current_params <- list(loss=loss_function,optimizer=opt,regularization=reg,regularization_weights=reg_weights,
                                     learning_rate=parameters$learning_rate,epochs=epoch,batch_size=parameters$batch_size,
                                     adam_params=ap)
              
              current_model <- CWAR(class_rules,train_trans_rules,train_trans_labels,validation_trans_rules,validation_trans_labels,
                            current_params,verbosity,logging)
              if(logging>0) {
                histories[[param_index]] <- current_model$history
                max_y <- max(unlist(current_model$history$accuracy),unlist(current_model$history$validation_accuracy))
              }
              if(is_better_model(current_model$validation_accuracy,current_model$num_rules,best_model$validation_accuracy,best_model$num_rules)) {
                best_model <- current_model
              }
              if(length(tested_models) < param_index) {
                  tested_models[[param_index]] <- list(val_acc=c(current_model$validation_accuracy),
                                                       rule_size=c(current_model$num_rules))
              } else {
                  tested_models[[param_index]]$val_acc <- c(tested_models[[param_index]]$val_acc,current_model$validation_accuracy)
                  tested_models[[param_index]]$rule_size <- c(tested_models[[param_index]]$rule_size,current_model$num_rules)
              }
              param_index <- param_index + 1
            }
          }
        }
      }
    }
  }
  if(logging>0) {
    plot(1:epoch,histories[[1]]$loss,log='y',ylim=(c(min(unlist(histories[[1]]$loss))/2,max(unlist(histories[[1]]$loss)))))
    if(param_index>2) {
      for(j in 2:param_index-1) {
        points(1:epoch,histories[[j]]$loss,log='y',col=j)
      }
      plot(1:epoch,histories[[1]]$rules,log='y',ylim=(c(min(unlist(histories[[1]]$rules))/2,max(unlist(histories[[1]]$rules)))))
      for(j in 2:param_index-1) {
        points(1:epoch,histories[[j]]$rules,log='y',col=j)
      }
      plot(1:epoch,histories[[1]]$accuracy)
      for(j in 2:param_index-1) {
        points(1:epoch,histories[[j]]$accuracy,log='y',col=j)
      }
      plot(1:epoch,histories[[1]]$validation_accuracy)
      for(j in 2:param_index-1) {
        points(1:epoch,histories[[j]]$validation_accuracy,log='y',col=j)
      }
    }
    
  }
  #print(paste('gridsearched',param_index,'params'))
  if(history) {
    #FLAG
    return(list(best=best_model,history=tested_models))
  } else {
    return(list(best=best_model))
  }
}

evaluate_model_accuracy <- function(model,trans_rules,trans_labels,batching=T,batch_size=16) {
  graph <- tf$get_default_graph()
  sess <- tf$get_default_session()
  accuracy <- graph$get_tensor_by_name('accuracy-tensor:0')
  acc <<- 0
  if(batching) {
    batches <- get_batches(dim(trans_rules)[1],batch_size)
    for(batch in batches) {
      acc <- acc + accuracy$eval(feed_dict=dict('C-tensor:0'=model$class_rules,
                                                'T-tensor:0'=get_batch(trans_rules,batch),
                                                'y-tensor:0'=get_batch(trans_labels,batch)),session=sess)
    }
    return(acc/length(batches))
  } else {
    return(accuracy$eval(feed_dict=dict('C-tensor:0'=model$class_rules,'T-tensor:0'=get_batch(trans_rules),
                                        'y-tensor:0'=get_batch(trans_labels),session=sess)))
  }
}

evaluate_baseline_accuracy <- function(trans_rules,trans_labels,class_rules,verbosity) {
  guesses <- apply(trans_rules%*%class_rules,1,function(x) which.max(x))
  labels <- apply(trans_labels,1,function(x) which.max(x))
  acc <- sum(guesses==labels)/length(guesses)
  if(verbosity>0) {
    print(paste('Baseline train acc:',train_acc,'Baseline test acc:',test_acc))
  }
  return(acc)
}

evaluate_rf_accuracy <- function(train_data,test_data,class_column,time=F) {
  target <- formula(paste0(class_column,'~.'))
  if(sum(count_na(train_data))==0) {
    clf <- randomForest(target,data=train_data)
    pred <- predict(clf,test_data)
  } else {
    #imputed_train_data <- rfImpute(target, train_data)
    #imputed_test_data <- rfImpute(target, test_data)
    imputed_train_data <- missForest(train_data)$ximp
    imputed_test_data <- missForest(test_data)$ximp
    clf <- randomForest(target,data=imputed_train_data)
    pred <- predict(clf,imputed_test_data)
  }
  acc <- sum(pred==test_data[[class_column]])/length(pred)
  if(time) {
    start_time <- proc.time()[1]
    for(j in 1:500) {
      predict(clf,test_data)
    }
    print(paste('rf time',proc.time()[1]-start_time))
  }
  return(acc)
}

run_nested_cross_validation <- function(name,raw_data,class_column,support,confidence,parameters,verbosity,logging,maxlength=10,
                                        compare_rf=T,compare_model=T,compare_time=F,fold_cutoff=10) {
  num_transactions <- dim(raw_data)[1]
  print(paste('Running cross validation for',name,'data with',num_transactions,'transactions total'))
  folds <- generate_folds(raw_data)
  
  ruleset_size_list <- vector('list',length(folds))
  test_accuracy_list <- vector('list',length(folds))
  baseline_test_accuracy_list <- vector('list',length(folds))
  rf_test_accuracy_list <- vector('list',length(folds))
  
  test_accuracy_list <- c()
  ruleset_size_list <- c()
  models <- list()
  baseline_test_accuracy_list <- c()
  rf_test_accuracy_list <- c()
  for(i in 1:length(folds)) {
    if(i>fold_cutoff) {
      #return(T)
      next
    }
    test_indices <- unlist(folds[i])
    train_indices <- setdiff(1:num_transactions,test_indices)
    validation_indices <- sample(train_indices,length(train_indices)/5)
    train_indices <- setdiff(train_indices,validation_indices)
    if(verbosity>-1) {
      print(paste('Running',name,'data on fold',i,'with',length(train_indices),'train transactions',
                  length(validation_indices),'val transactions, and',length(test_indices),'test transactions'))
    }
    if(compare_rf) {
      rf_test_acc <- evaluate_rf_accuracy(raw_data[train_indices,],raw_data[test_indices,],class_column)
      rf_test_accuracy_list[[i]] <- rf_test_acc
    }
    if(!compare_model) {
      next
    }
    rules <- mine_rules_nested(raw_data,train_indices,class_column,support,confidence,maxlength=maxlength)
    if(verbosity>-1) {
      print(paste('Mined',length(rules),'rules'))
    }
    class_rules <- find_rules_per_class(rules)
    #rm(rules)
    
    train_trans_rules <- find_rules_per_transaction(rules,as(raw_data[train_indices,],'transactions')) #THIS IS THE BOTTLENECK
    train_trans_labels <- generate_labels(raw_data[train_indices,],class_column)
    validation_trans_rules <- find_rules_per_transaction(rules,as(raw_data[validation_indices,],'transactions'))
    validation_trans_labels <- generate_labels(raw_data[validation_indices,],class_column)
    
    
    grid_search_results <- run_grid_search(class_rules,train_trans_rules,train_trans_labels,validation_trans_rules,validation_trans_labels,
                                  parameters,verbosity,logging,history=T,tested_models=models)
                                  #parameters,verbosity,logging,history=F)
    best_model <- grid_search_results$best
    if(logging>0) {
      #plot(quality(rules)$confidence,rowSums(best_model$weights))
    }
    models <- grid_search_results$history
    if(verbosity>0 & i==10) {
      print(models)
    }
    
    rm(train_trans_rules,train_trans_labels,validation_trans_rules,validation_trans_labels)
    
    test_trans_rules <- find_rules_per_transaction(rules,as(raw_data[test_indices,],'transactions'))
    test_trans_labels <- generate_labels(raw_data[test_indices,],class_column)
    
    #FLAG
    #model_test_acc <- evaluate_model_accuracy(best_model,test_trans_rules,test_trans_labels,batch_size=parameters$batch_size)
    model_test_acc <- evaluate_baseline_accuracy(test_trans_rules,test_trans_labels,best_model$weights,verbosity)
    test_accuracy_list[[i]] <- model_test_acc
    ruleset_size_list[[i]] <- best_model$num_rules
    
    baseline_test_acc <- evaluate_baseline_accuracy(test_trans_rules,test_trans_labels,class_rules,verbosity=verbosity)
    baseline_test_accuracy_list[[i]] <- baseline_test_acc
    
    if(verbosity>-1) {
      print(paste('model test acc',model_test_acc,'baseline test acc',baseline_test_acc))
      if(compare_rf) {
        print(paste('rf test acc',rf_test_acc))
      }
      print(paste('model rules',best_model$num_rules))
    }
    #cba_test_acc <- evaluate_cba_accuracy()
  }
  if(compare_time) {
    target <- formula(paste0(class_column,'~.'))
    classifier <- make_classifier(rules,best_model$weights,target)
    test_data <- raw_data[test_indices,]
    start_time <- proc.time()[1]
    for(j in 1:500) {
      predict(classifier,test_data)
    }
    print(paste('classifier time',proc.time()[1]-start_time))
    evaluate_rf_accuracy(raw_data[train_indices,],raw_data[test_indices,],class_column,T)
  }
  print(paste('Average model test accuracy:',mean(test_accuracy_list)))
  print(paste('Average model ruleset size:',mean(ruleset_size_list)))
  print(paste('Average baseline test accuracy:',mean(baseline_test_accuracy_list)))
  print(paste('Average rf test accuracy:',mean(rf_test_accuracy_list)))
}


adam_p <- list(learning_rate=0.010,beta1=0.9,beta2=0.999,epsilon=1e-08)
adam_p2 <- list(learning_rate=0.020,beta1=0.9,beta2=0.999,epsilon=1e-08)
adadelta_p <- list(learning_rate=0.001,rho=0.95,epsilon=1e-08)
reg_weights_0 <- list(l1=0.00005,l2=0.0001)
reg_weights <- list(l1=0.0001,l2=0.0001)
reg_weights_2 <- list(l1=0.001,l2=0.0001)
reg_weights_3 <- list(l1=0.005,l2=0.0001)
reg_weights_4 <- list(l1=0.007,l2=0.0001)
gridsearch_p <- list(epochs=c(8),learning_rate='purple',batch_size=16,loss=c('cross'),optimizer=c('adam'),
                     adam_params=list(adam_p2),
                     regularization=c('l1'),regularization_weights=list(reg_weights,reg_weights_2,reg_weights_3,
                                                                        reg_weights_4))#,early_stop='test_acc',patience=4L)
labor_p <- list(epochs=c(8),learning_rate='purple',batch_size=4,loss=c('cross'),optimizer=c('adam'),
                     adam_params=list(adam_p2),
                     regularization=c('l1'),regularization_weights=list(reg_weights,reg_weights_2,reg_weights_3,
                                                                        reg_weights_4))#,early_stop='test_acc',patience=4L)
run_nested_cross_validation('Anneal',prepare_anneal(),'V39',0.1,0.75,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation('Austral',prepare_austral(),'V15',0.1,0.75,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation('Auto',prepare_auto(),'V26',0.3,0.6,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation('Breast',prepare_bc(),'V11',0.01,0.5,gridsearch_p,verbosity=0,logging=1,compare_rf = T,compare_time=F)

run_nested_cross_validation('CRX',prepare_crx(),'V16',0.01,0.92,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation('Cleve',prepare_cleve(),'V14',0.01,0.7,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation('German',prepare_german(),'V21',0.04,0.85,gridsearch_p,verbosity=0,logging=0,compare_rf=T,compare_time=F)

run_nested_cross_validation('Glass',prepare_glass(),'V11',0.01,0.5,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation('Heart',prepare_heart(),'V14',0.01,0.85,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation('Hepatic',prepare_hepatic(),'V20',0.25,0.75,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation('Horse',prepare_horse(),'V24',0.2,0.7,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation('Hypo',prepare_hypo(),'V24',0.2,0.7,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation("Iono",prepare_iono(),'V35',0.35,0.6,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation('Iris',prepare_iris(),'V4',0.01,0.5,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation("Labor",prepare_labor(),'V17',0.21,0.75,labor_p,verbosity=0,logging=0,compare_rf=F,compare_time=F)

run_nested_cross_validation("LED7",prepare_led7(),'V8',0.01,0.5,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation("Lymph",prepare_lymph(),'V19',0.199,0.75,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation("Pima",prepare_pima(),'V9',0.01,0.5,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation("Sick",prepare_sick(),'V30',0.32,0.75,gridsearch_p,verbosity=0,logging=0,compare_rf=T,compare_time=T,maxlength=6)

run_nested_cross_validation("Sonar",prepare_sonar(),'V61',0.1,0.7,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation("Tic",prepare_tic(),'V10',0.01,0.5,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation("Vehicle",prepare_vehicle(),'V19',0.05,0.75,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation("Waveform",prepare_waveform(),'V22',0.01,0.75,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation("Wine",prepare_wine(),'V13',0.01,0.75,gridsearch_p,verbosity=0,logging=0,compare_rf=T,compare_time=F)

run_nested_cross_validation("Zoo",prepare_zoo(),'V18',0.01,0.5,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=F)

run_nested_cross_validation("Census",prepare_census(),'V42',0.15,0.65,gridsearch_p,verbosity=0,logging=1,compare_rf=F,compare_time=F,
                            fold_cutoff=4,maxlen=5)

run_nested_cross_validation("Poker",prepare_poker(),'V11',0.02,0.54,gridsearch_p,verbosity=0,logging=1,compare_rf=F,compare_model=T,
                            compare_time=F,fold_cutoff=4,maxlen=4)