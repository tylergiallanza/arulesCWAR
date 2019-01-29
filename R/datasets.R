library('devtools')
#install_github('ianjjohnson/arulesCBA')
#install('C:\\Users\\Tylergiallanza\\Downloads\\arulesCBA-master\\arulesCBA-master')
library(arulesCBA)
library(randomForest)
library(tensorflow)

count_na <- function(df) {
  sapply(df, function(y) sum(length(which(is.na(y)))))
}

prepare_vote <- function() {
  vote_data <- read.csv(url('https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data'),
                          header=FALSE,sep=',', na.strings = '?')
  
  vote_data$cls <- factor(vote_data[,'V1']) #class variable
  vote_data[,'V1'] <- NULL
  vote_data <- discretizeDF.supervised('cls~.',vote_data)
  #vote_data <- as(vote_data,'transactions')
  vote_data
}

prepare_mush <- function() {
  mush_data <- read.csv(url('https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'),
                          header=FALSE,sep=',', na.strings = '?')
  
  mush_data$cls <- factor(mush_data[,'V1']) #class variable
  mush_data[,'V1'] <- NULL
  mush_data <- discretizeDF.supervised('cls~.',mush_data)
  #vote_data <- as(vote_data,'transactions')
  mush_data
}

prepare_vote_params <- function() {
  reg_params <- list(l1=0.005)
  list(support=0.1,confidence=1,regularization="l1",learning_rate=0.05,epoch=200,regularization_weights=reg_params)
  #supports in c(0.05, 0.1, 0.2, 0.25)
  #confs in c(.5,.75,.9,1)
}

prepare_mush_params <- function() {
  reg_params <- list(l1=0.005)
  list(support=0.1,confidence=1,regularization="l1",learning_rate=0.05,epoch=50,regularization_weights=reg_params,
       batch_size=128)
  #support in c(0.2, 0.25, 0.3)
  #confs in c(.5, .75, .9, 1)
}

run_inner_crossval <- function(formula, trans, cwar_params, crossval_index = 0) {
  split_point <- round(0.9*length(trans))
  train_trans <- trans[1:split_point]
  test_trans <- trans[split_point:length(trans)]
  accs <- list()
  models <- list()
  model_index <- 1
  print("Running grid search.")
  #for(support in c(0.05, 0.1, 0.2, 0.25)) {
  for(support in c(0.2, 0.25)){#, 0.3)){
    for(confidence in c(.5, .75, .9, 1)) {
      print("    Running grid iteration...")
      temp_params <- cwar_params
      temp_params$support <- support
      temp_params$confidence <- confidence
      model <- CWAR(formula, trans, temp_params, 0)
      y_true <- transactions_to_labels(formula,test_trans)
      y_pred <- predict(model, test_trans)
      acc <- sum(y_true==y_pred)/length(y_true)
      accs[[model_index]] <- acc
      models[[model_index]] <- model
      model_index <- model_index + 1
    }
  }
  model_params <- unlist(lapply(models, function(x) x$params))
  model_params <- list()
  for(model_index in 1:length(models)) {
    model_params[[model_index]] <- models[[model_index]]$params
  }
  save(accs,model_params,file=paste0('models-',crossval_index,'.RData'))
  
  return(models[[which.max(accs)]])
}

find_rf_params <- function(formula, data) {
  split_point <- round(0.9*length(data))
  train_data <- data[1:split_point,]
  test_data <- data[split_point:length(data),]
  base_mtry <- floor(sqrt(ncol(train_data)))
  best_acc <- 0
  best_model <- NULL
  for(ntree in c(500,1000,2000)) {
    for(mtry in c(floor(base_mtry/2),base_mtry,base_mtry*2)) {
      model <- randomForest(formula, train_data, ntree = ntree, mtry = mtry, na.action=na.roughfix)
      y_pred <- predict(model, na.roughfix(test_data))
      y_true <- transactions_to_labels(formula,as(test_data,'transactions'))
      acc <- sum(y_true==y_pred)/length(y_true)
      if(acc > best_acc) {
        best_acc <- acc
        best_model <- model
      }
    }
  }
  return(best_model)
}


run_crossval <- function(formula, data, cwar_params, crossval_index = 0, run_model = F, run_rf = F, run_cba = F, run_rcar = F) {
  trans <- as(data, 'transactions')
  test_length <- floor(0.1*length(trans))
  test_start <- crossval_index*test_length
  test_indices <- 1:test_length+test_start
  train_data <- data[-test_indices,]
  train_trans <- trans[-test_indices]
  test_data <- data[test_indices,]
  test_trans <- trans[test_indices]
  #model <- CWAR(formula, trans, cwar_params, 1)
  y_true <- transactions_to_labels(formula,test_trans)
  return_model <- NULL
  if(run_model) {
    model <- run_inner_crossval(formula, train_trans, cwar_params, crossval_index)
    return_model <- model
    y_pred <- predict(model, test_trans)
    plot(1:cwar_params$epoch,model$history$loss)
    plot(1:cwar_params$epoch,model$history$accuracy)
    plot(1:cwar_params$epoch,model$history$rules)
    print('model results')
    print(confusionMatrix(y_pred,y_true))
  }
  if(run_rf) {
    rf_model <- find_rf_params(formula, train_data)
    y_pred <- predict(rf_model, na.roughfix(test_data))
    print('rf results')
    print(confusionMatrix(y_pred,y_true))
    print(rf_model)
  }
  if(run_cba) {
    #supp <- model$support
    #conf <- model$confidence
    supp <- 0.05
    conf <- 0.5
    cba_model <- CBA(formula, train_data, support = supp, confidence = conf)
    y_pred <- predict(cba_model, test_data)
    print('cba results')
    print(confusionMatrix(y_pred,y_true))
    if(!run_model) {
      return_model <- cba_model
    }
  }
  if(run_rcar) {
    supp <- model$support
    conf <- model$confidence
    rcar_model <- RCAR(formula, train_data, support = supp, confidence = conf)
    y_pred <- predict(rcar_model, test_data)
    print(confusionMatrix(y_pred,y_true))
  }
  
  return(return_model)
}
if(F) {
#good
prepare_anneal <- function() {
  anneal_data <- read.csv(url('https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.data'),
                          header=FALSE,sep=',')
  
  anneal_data[['V1']] <- NULL
  anneal_data$V39 <- factor(anneal_data[,'V39']) #class variable
  anneal_data <- discretizeDF.supervised('V39~.',anneal_data)
  
  
  anneal_data
}

#good
prepare_austral <- function() {
  austral_data <- read.csv(url('http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat'),
                           header=FALSE,sep=' ')
  austral_data$V15 <- factor(austral_data[,'V15']) #CLASS VARIABLE
  austral_data <- discretizeDF.supervised('V15~.',austral_data)
  austral_data
}

#good
prepare_auto <- function() {
  auto_data <- read.csv(url('https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'),
                        header=FALSE,sep=',',na.strings='?')
  
  temp <- factor(auto_data[['V1']])
  auto_data[['V2']] <- NULL
  auto_data$V26 <- temp #class variable
  auto_data <- discretizeDF.supervised('V26~.',auto_data)
  
  auto_data
}

#good
prepare_bc <- function() {
  bc_data <- read.csv(url('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'),header=FALSE)
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

#good
prepare_crx <- function() {
  crx_data <- read.csv(url('https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'),header=FALSE,
                       na.strings='?')
  crx_data[,'V16'] <- factor(crx_data[,'V16']) #CLASS VARIABLE
  
  crx_data <- discretizeDF.supervised('V16~.',crx_data)
  crx_data
}

#good
prepare_cleve <- function() {
  cleve_data <- read.csv(url('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'),
                         header=F,sep=',')
                   
  change_indices <- c(which(cleve_data$V14!='0'))
  cleve_data[change_indices,'V14'] <- 1
  cleve_data[,'V14'] <- factor(cleve_data[,'V14'])
  cleve_data <- discretizeDF.supervised('V14~.',cleve_data)
  cleve_data
}

#good
prepare_german <- function() {
  german_data <- read.csv(url('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'),header=FALSE,sep=' ')
  german_data$V21 <- factor(german_data[,'V21']) #Class variable
  german_data <- discretizeDF.supervised('V21~.',german_data)
  german_data
}

#good
prepare_glass <- function() {
  glass_data <- read.csv(url('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'),header=FALSE,sep=',')
  glass_data$V1 <- NULL
  glass_data$V11 <- factor(glass_data[,'V11']) #Class variable
  glass_data <- discretizeDF.supervised('V11~.',glass_data)
  glass_data
}

#good
prepare_heart <- function() {
  heart_data <- read.table(url('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat'),
                           sep=' ',header=FALSE)
  heart_data[,'V14'] <- factor(heart_data[,'V14'])
  heart_data <- discretizeDF.supervised('V14~.',heart_data)
  heart_data
}

#good
prepare_hepatic <- function() {
  hepatic_data <- read.csv(url('https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data'),sep=',',header=FALSE)
  temp <- factor(hepatic_data[,'V1'])
  hepatic_data$V1 <- hepatic_data[,'V20']
  hepatic_data$V20 <- temp
  hepatic_data <- discretizeDF.supervised('V20~.',hepatic_data)
  hepatic_data
}

#good
prepare_horse <- function() {
  #horse_data <- read.table('~/Dropbox/School/CSE5393/sgd/horse.csv',
  #                         sep=',',header=FALSE)
  horse_data <- read.table('D:\\Dropbox\\School\\CSE5393\\sgd\\horse.csv',
                           sep=',',header=FALSE,na.strings='?')
  horse_data[,'V3'] <- NULL
  horse_data[,'V24'] <- factor(horse_data[,'V24'])
  horse_data <- discretizeDF.supervised('V24~.',horse_data)
  
  horse_data
}

#good
prepare_iono <- function() {
  iono_data <- read.table(url('https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'),
                           sep=',',header=FALSE)
  iono_data[,'V35'] <- factor(iono_data[,'V35'])
  iono_data <- discretizeDF.supervised('V35~.',iono_data)
  iono_data
}

#good
prepare_iris <- function() {
  iris_data <- read.table(url('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'),
                           sep=',',header=FALSE)
  iris_data[,'V5'] <- factor(iris_data[,'V5'])
  iris_data <- discretizeDF.supervised('V5~.',iris_data)
  iris_data
}

#good
prepare_labor <- function() {
  labor_data <- read.csv('~/Dropbox/School/CSE5393/sgd/labor.csv',sep=',',header=FALSE,na.strings='?')
  labor_data$V17 <- factor(labor_data$V17)
  
  labor_data <- discretizeDF.supervised('V17~.',labor_data)
  labor_data
}

#good
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

#good
prepare_lymph <- function() {
  lymph_data <- read.table(url('https://archive.ics.uci.edu/ml/machine-learning-databases/lymphography/lymphography.data'),
                           sep=',',header=FALSE)
  temp <- factor(lymph_data[,'V1'])
  lymph_data[,'V1'] <- lymph_data[,'V19']
  lymph_data[,'V19'] <- temp #class variable
  lymph_data <- discretizeDF.supervised('V19~.',lymph_data)
  lymph_data
}

#good
prepare_pima <- function() {
  pima_data <- read.csv(url('https://gist.githubusercontent.com/ktisha/c21e73a1bd1700294ef790c56c8aec1f/raw/819b69b5736821ccee93d05b51de0510bea00294/pima-indians-diabetes.csv'),
                        sep=',',header=FALSE,skip=9)
  pima_data[,'V9'] <- factor(pima_data[,'V9']) #class variable
  pima_data <- discretizeDF.supervised('V9~.',pima_data)
  
  pima_data
}

#good
prepare_sick <- function() {
  sick_data <- read.table(url('https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/sick.data'),
                           sep=',',header=FALSE)
  sick_data[,'V30'] <- as.character(levels(sick_data[,'V30']))[sick_data[,'V30']]
  class_labels <- unlist(strsplit(sick_data[,'V30'],"\\."))[1:length(sick_data[,'V30'])*2-1]
  sick_data[,'V30'] <- factor(class_labels)
  sick_data <- discretizeDF.supervised('V30~.',sick_data)
  sick_data
}

#good
prepare_sonar <- function() {
  sonar_data <- read.table(url('http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'),
                           sep=',',header=FALSE)
  sonar_data[,'V61'] <- factor(sonar_data[,'V61']) #class variable
  sonar_data <- discretizeDF.supervised('V61~.',sonar_data)
  
  sonar_data
}

#good
prepare_tic <- function() {
  tic_data <- read.table(url('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data'),
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

#good
prepare_wine <- function() {
  wine_data <- read.table(url('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'),
                           sep=',',header=FALSE)
  temp <- factor(wine_data[,'V1'])
  wine_data[,'V1'] <- wine_data[,'V13']
  wine_data[,'V13'] <- temp
  
  wine_data <- discretizeDF.supervised('V13~.',wine_data)
  wine_data
}

#good
prepare_waveform <- function() {
  waveform_data <- read.table('~/Dropbox/School/CSE5393/sgd/waveform.data',sep=',',header=F,na.strings='?')
  waveform_data$V22 <- factor(waveform_data$V22)
  
  waveform_data <- discretizeDF.supervised('V22~.',waveform_data)
  waveform_data
}

#good
prepare_vehicle <- function() {
  vehicle_data <- read.table('~/Dropbox/School/CSE5393/sgd/vehicle.csv',sep=',',header=F,strip.white=T,
                             na.strings = '?')
  vehicle_data$V19 <- factor(vehicle_data$V19)
  vehicle_data <- discretizeDF.supervised('V19~.',vehicle_data)
  vehicle_data
}

#good
prepare_zoo <- function() {
  zoo_data <- read.table(url('http://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data'),
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

process_data <- function(data, data_label, test_fold, class_column, sup, conf,maxlength=10) {
  #num_train_elems <- floor(dim(data)[1]*train_pct)
  #train_idx <- sample(seq_len(nrow(data)), size=num_train_elems)
  
  train_itemset <- data[-unlist(test_fold),]
  train_data <- as(train_itemset,'transactions')
  test_itemset <- data[unlist(test_fold),]
  test_data <- as(test_itemset,'transactions')
  data_rules <- apriori(train_data, parameter=list(support=sup, confidence = conf, maxlen=maxlength),
    appearance = list(rhs=grep(class_column, itemLabels(train_data), value = TRUE), default = "lhs"),
    control=list(verbose=F))
  train_test <- list()
  train_test[c('train','test','train_itemset','test_itemset','ruleset','class_column','label')] <- list(
    train_data,test_data,train_itemset,test_itemset,data_rules,class_column,data_label)
  train_test
}

mine_rules_nested <- function(data, data_indices, class_column, sup, conf,maxlength=10) {
  train_data <- as(data[data_indices,],'transactions')
  data_rules <- apriori(train_data, parameter=list(support=sup, confidence = conf, maxlen=maxlength),
    appearance = list(rhs=grep(class_column, itemLabels(train_data), value = TRUE), default = "lhs"),
    control=list(verbose=F))
  data_rules
}


find_rules_per_class <- function(rules,method=2) {
  if(method==1) {
    classes <- unique(rhs(rules))
    class_list <- list()
    for(i in 1:length(classes)) {
      class_list[length(class_list)+1] <- list(which(is.subset(rhs(rules),classes[i])))
    }
    class_list
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

#PRECONDITION: every transaction has at least 1 rule, b/c there is a default rule that guesses the majority class
#params list: (epoch,eta,batch_size,loss=['mse','cross'],optimizer=['sgd','adam','adadelta'],
#              adam_params=[l_r,b1,b2,eps],adadelta_params=[l_r,rho,eps],
#             regularization=[l1,l2],regularization_weights=[l1,l2])
build_and_run <- function(t_in,y_in,t_in_test,y_in_test,c_in,params,verbosity=0,logging=0) {
  best_epoch <- 1
  train_accs <- c()
  test_accs <- c()
  train_loss <- c()
  test_loss <- c()
  tf$reset_default_graph() 
  with(tf$Session() %as% sess,{
  w_in <- gen_starting_weights(c_in)
  num_rules <- dim(c_in)[1]
  num_classes <- dim(c_in)[2]
  
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
  
  yhat <- tf$matmul(T_,W,a_is_sparse = T, b_is_sparse = T,name='yhat-tensor')
  
  if(params$loss=='mse') {
    loss <- tf$losses$mean_squared_error(y_,yhat)
  } else if(params$loss=='cross') {
    loss <- tf$losses$softmax_cross_entropy(y_,yhat)
  }
  
  if(params$regularization=='l1') {
    regularizer <- tf$scalar_mul(params$regularization_weights$l1,tf$reduce_sum(tf$abs(W)))
    loss <- tf$add(loss,regularizer)
  } else if(params$regularization=='l2') {
    regularizer <- tf$scalar_mul(params$regularization_weights$l1,tf$reduce_sum(tf$square(W)))
    loss <- tf$reduce_mean(tf$add(loss,regularizer))
  } else if(params$regularization=='elastic') {
    l1 <- tf$scalar_mul(params$regularization_weights$l1,tf$reduce_sum(tf$abs(W)))
    l2 <- tf$scalar_mul(params$regularization_weights$l1,tf$reduce_sum(tf$square(W)))
    loss <- tf$reduce_mean(tf$add(tf$add(loss,l1),l2))
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
  #print(paste(C_,y_,T_,W,loss,optimizer))
  train_step <- optimizer$minimize(loss)
  
  sess$run(tf$global_variables_initializer())
  for(i in 1:params$epoch) {
    batches <- get_batches(dim(t_in)[1],params$batch_size)
    for(batch in batches) {
      current_batch <- get_batch(t_in,batch)
      #print(tail(sort( sapply(ls(),function(x){object.size(get(x))})) ))
      sess$run(train_step,
             feed_dict = dict(C_=c_in,T_=current_batch,y_=get_batch(y_in,batch)))
      rm(current_batch)
    }
    if('early_stop' %in% names(params) | logging==1) {
      train_accs <- c(train_accs,eval_accuracy(c_in,t_in,y_in,sess,T,params$batch_size))
      train_loss <- c(train_loss,eval_loss(loss,c_in,t_in,y_in,sess,T,params$batch_size))
      test_accs <- c(test_accs,eval_accuracy(c_in,t_in_test,y_in_test,sess,F))
      test_loss <- c(test_loss,eval_loss(loss,c_in,t_in_test,y_in_test,sess,F))
      saver$save(sess,paste0('/tmp/model-',i,'.ckpt'))
    }
    
    if('early_stop' %in% names(params)) {
      if(params$early_stop == 'train_acc') {
        eps <- 0.0001
        if(train_accs[i]+eps > train_accs[best_epoch]) {
          best_epoch <- i
        } else if(i>best_epoch+params$patience) {
          saver$restore(sess,paste0('/tmp/model-',best_epoch,'.ckpt'))
          print(paste('Restored model to epoch',best_epoch))
          break
        }
      } else if(params$early_stop == 'test_acc') {
        eps <- 0.0001
        if(test_accs[i]+eps > test_accs[best_epoch]) {
          best_epoch <- i
        } else if(i>best_epoch+params$patience) {
          saver$restore(sess,paste0('/tmp/model-',best_epoch,'.ckpt'))
          print(paste('Restored model to epoch',best_epoch))
          break
        }
      }
    }
    
    if(verbosity>1) {
      print(paste(i,'train acc:',eval_accuracy(c_in,t_in,y_in,sess,T,params$batch_size),'train loss:',eval_loss(loss,c_in,t_in,y_in,sess,T,params$batch_size)))
      print(paste('   test acc:',eval_accuracy(c_in,t_in_test,y_in_test,sess,F),
                  'test loss:',eval_loss(loss,c_in,t_in_test,y_in_test,sess,F)))
      num_rules <- sess$run(tf$count_nonzero(tf$nn$relu(W)),feed_dict=dict(C_=c_in)) #ADD RELU
      print(paste('    num rules:',num_rules))
    }
  }
  train_acc <- eval_accuracy(c_in,t_in,y_in,sess,T,params$batch_size)
  test_acc <- eval_accuracy(c_in,t_in_test,y_in_test,sess,F)
  num_rules <- sess$run(tf$count_nonzero(tf$nn$relu(W)),feed_dict=dict(C_=c_in)) #ADD RELU
  if(verbosity>0) {
    print(paste('Train acc:',train_acc,'Test acc:',test_acc))
    #print(sess$run(W,
    #         feed_dict = dict(C_=c_in,T_=t_in[batch,],y_=y_in[batch,])))
  }
  if(logging==1) {
    png('train_accs.png')
    plot(train_accs)
    dev.off()
    png('test_accs.png')
    plot(test_accs)
    dev.off()
    png('train_loss.png')
    plot(train_loss)
    dev.off()
    png('test_loss.png')
    plot(test_loss)
    dev.off()
  }
  #print(tail(sort(sapply(ls(),function(x){object.size(get(x))})) ))
  return(c(train_acc,test_acc,num_rules))
  
  })
}

CWAR <- function(class_rules,train_trans_rules,train_trans_labels,validation_trans_rules,validation_trans_labels,
                 params,verbosity=0,logging=0) {
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
    yhat <- tf$matmul(T_,W,a_is_sparse = T, b_is_sparse = T,name='yhat-tensor')
    
    if(params$loss=='mse') {
      loss <- tf$losses$mean_squared_error(y_,yhat)
    } else if(params$loss=='cross') {
      loss <- tf$losses$softmax_cross_entropy(y_,yhat)
    }
    
    if(params$regularization=='l1') {
      regularizer <- tf$scalar_mul(params$regularization_weights$l1,tf$reduce_sum(tf$abs(W)))
      loss <- tf$add(loss,regularizer)
    } else if(params$regularization=='l2') {
      regularizer <- tf$scalar_mul(params$regularization_weights$l1,tf$reduce_sum(tf$square(W)))
      loss <- tf$reduce_mean(tf$add(loss,regularizer))
    } else if(params$regularization=='elastic') {
      l1 <- tf$scalar_mul(params$regularization_weights$l1,tf$reduce_sum(tf$abs(W)))
      l2 <- tf$scalar_mul(params$regularization_weights$l1,tf$reduce_sum(tf$square(W)))
      loss <- tf$reduce_mean(tf$add(tf$add(loss,l1),l2))
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

    train_step <- optimizer$minimize(loss)#problem line
    
    sess$run(tf$global_variables_initializer())
    for(i in 1:params$epoch) {
      batches <- get_batches(dim(train_trans_rules)[1],params$batch_size)
      for(batch in batches) {
        current_batch <- get_batch(train_trans_rules,batch)
        sess$run(train_step,
               feed_dict = dict(C_=class_rules,T_=current_batch,y_=get_batch(train_trans_labels,batch)))
        rm(current_batch)
      }
    }
    model <- list(model=NULL,class_rules=class_rules,validation_accuracy=0,num_rules=0)
    val_acc <- evaluate_model_accuracy(model,validation_trans_rules,validation_trans_labels,batch_size=params$batch_size)
    n_rules <- sess$run(tf$count_nonzero(tf$nn$relu(W)),feed_dict=dict(C_=class_rules)) #ADD RELU
    model$validation_accuracy <- val_acc
    model$num_rules <- n_rules
    model$weights <- sess$run(W,feed_dict=dict(C_=class_rules)) #combine this with n_rules above
    return(model)
  })
}

run_grid_search <- function(class_rules,train_trans_rules,train_trans_labels,validation_trans_rules,validation_trans_labels,
                              parameters,verbosity,logging,history=F,tested_models=list()) {
  best_model <<- list(validation_accuracy=0)
  param_index <<- 1
  for(loss_function in parameters$loss) {
    for(opt in parameters$optimizer) {
      for(reg in parameters$regularization) {
        for(reg_weights in parameters$regularization_weights) {
          for(l_rate in parameters$learning_rate) {
            for(epoch in parameters$epochs) {
              current_params <- list(loss=loss_function,optimizer=opt,regularization=reg,regularization_weights=reg_weights,
                                     learning_rate=l_rate,epochs=epoch,batch_size=parameters$batch_size,
                                     adam_params=parameters$adam_params)
              
              current_model <- CWAR(class_rules,train_trans_rules,train_trans_labels,validation_trans_rules,validation_trans_labels,
                            current_params,verbosity,logging)
              if(current_model$validation_accuracy > best_model$validation_accuracy) {
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
  #print(paste('gridsearched',param_index,'params'))
  if(history) {
    #FLAG
    return(list(best=best_model,history=tested_models))
  } else {
    return(list(best=best_model))
  }
}


#THIS IS THE BOTTLENECK - IT REQUIRES THE WHOLE T NOT JUST THE BATCHES
eval_accuracy <- function(c_in,t_in,y_in,sess,batching=T,batch_size=0) {
  graph <- tf$get_default_graph()
  y_ <- graph$get_tensor_by_name('y-tensor:0')
  yhat <- graph$get_tensor_by_name('yhat-tensor:0')
  correct_prediction <- tf$equal(tf$argmax(yhat,1L),tf$argmax(y_,1L))
  accuracy <- tf$reduce_mean(tf$cast(correct_prediction,tf$float32))
  acc <<- 0
  if(batching) {
    batches <- get_batches(dim(t_in)[1],batch_size)
    for(batch in batches) {
      acc <- acc + accuracy$eval(feed_dict=dict('C-tensor:0'=c_in,'T-tensor:0'=get_batch(t_in,batch),'y-tensor:0'=get_batch(y_in,batch)),session=sess)
    }
  } else {
    return(accuracy$eval(feed_dict=dict('C-tensor:0'=c_in,'T-tensor:0'=get_batch(t_in),'y-tensor:0'=y_in),session=sess))
  }
  return(acc/length(batches))
}

eval_loss <- function(loss,c_in,t_in,y_in,sess,batching=T,batch_size=0) {
  if(batching) {
    batches <- get_batches(dim(t_in)[1],batch_size)
    lss <<- 0
    for(batch in batches) {
      lss <- lss + loss$eval(feed_dict=dict('C-tensor:0'=c_in,'T-tensor:0'=get_batch(t_in,batch),'y-tensor:0'=get_batch(y_in,batch)),session=sess)
    }
  } else {
    return(loss$eval(feed_dict=dict('C-tensor:0'=c_in,'T-tensor:0'=get_batch(t_in),'y-tensor:0'=y_in),session=sess))
  }
  return(lss/length(batches))
}

get_baseline_accuracy <- function(trans_rules,trans_rules_test,y_in,y_in_test,w_in,verbosity=0) {
  train_guesses <- apply(trans_rules%*%w_in,1,function(x) which.max(x))
  train_labels <- apply(y_in,1,function(x) which.max(x))
  train_acc <- sum(train_guesses==train_labels)/length(train_guesses)
  test_guesses <- apply(trans_rules_test%*%w_in,1,function(x) which.max(x))
  test_labels <- apply(y_in_test,1,function(x) which.max(x))
  test_acc <- sum(test_guesses==test_labels)/length(test_guesses)
  if(verbosity>0) {
    print(paste('Baseline train acc:',train_acc,'Baseline test acc:',test_acc))
  }
  return(c(train_acc,test_acc,dim(w_in)[1]))
}

evaluate_model_accuracy <- function(model,trans_rules,trans_labels,batching=T,batch_size=16) {
  graph <- tf$get_default_graph()
  sess <- tf$get_default_session()
  y_ <- graph$get_tensor_by_name('y-tensor:0')
  yhat <- graph$get_tensor_by_name('yhat-tensor:0')
  correct_prediction <- tf$equal(tf$argmax(yhat,1L),tf$argmax(y_,1L))
  accuracy <- tf$reduce_mean(tf$cast(correct_prediction,tf$float32))
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
    return(accuracy$eval(feed_dict=dict('T-tensor:0'=get_batch(trans_rules),'y-tensor:0'=get_batch(trans_labels),session=sess)))
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
  clf <- randomForest(target,data=train_data)
  pred <- predict(clf,test_data)
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
get_cba_accuracy <- function(dataset,support,confidence,maxlength=10) {
  if(maxlength<10) {
    return(c(0,0,0))
  }
  data_target <- paste(dataset$class_column,'~.',sep='')
  cba <- CBA(data_target, data=dataset$train_itemset, parameter=list(maxlen=maxlength),
             support=support,confidence=confidence)
  num_rules <- length(cba$rules)
  train_prediction <- predict(cba,dataset$train_itemset)
  test_prediction <- predict(cba,dataset$test_itemset)
  train_acc <- sum(train_prediction==dataset$train_itemset[[dataset$class_column]])/length(dataset$train_itemset[[dataset$class_column]])
  test_acc <- sum(test_prediction==dataset$test_itemset[[dataset$class_column]])/length(dataset$test_itemset[[dataset$class_column]])
  return(c(train_acc,test_acc,num_rules))
}

run_cross_validation <- function(name,raw_data,class_column,support,confidence,parameters,verbosity,logging,maxlength=10) {
  print(paste('Running cross validation for',name,'data with',dim(raw_data)[1],'transactions total'))
  folds <- generate_folds(raw_data)
  cba_train <- c()
  cba_train_rules <- c()
  cba_test <- c()
  baseline_train <- c()
  baseline_train_rules <- c()
  baseline_test <- c()
  acc_train <- c()
  train_rules <- c()
  acc_test <- c()
  for(i in 1:10) {
    if(i>1) {
      #return(T)
      next
    }
    if(verbosity>-1) {
      print(paste('Running',name,'data on fold',i,'with',length(unlist(folds[i])),'test transactions'))
    }
    data <- process_data(raw_data,name,folds[i],class_column,support,confidence,maxlength=maxlength)
    #print(paste('Finished process data method in',as.numeric(t[2])))
    if(verbosity>-1) {
      print(paste('Mined',length(data$ruleset),'rules'))
    }
    class_rules <- find_rules_per_class(data$ruleset)
    trans_rules <- find_rules_per_transaction(data$ruleset,data$train) #THIS IS THE BOTTLENECK
    trans_rules_test <- find_rules_per_transaction(data$ruleset,data$test)#THIS IS A LIE!!!
    trans_labels <- generate_labels(data$train_itemset,data$class_column)
    trans_labels_test <- generate_labels(data$test_itemset,data$class_column)
    
    baseline_accs <- get_baseline_accuracy(trans_rules,trans_rules_test,trans_labels,
                                           trans_labels_test,class_rules,verbosity=verbosity)
    cba_results <- get_cba_accuracy(data,support,confidence,maxlength=maxlength)
    baseline_train <- c(baseline_train,baseline_accs[1])
    baseline_test <- c(baseline_test,baseline_accs[2])
    baseline_train_rules <- c(baseline_train_rules,baseline_accs[3])
    cba_train <- c(cba_train,cba_results[1])
    cba_test <- c(cba_test,cba_results[2])
    cba_train_rules <- c(cba_train_rules,cba_results[3])
    accs <- build_and_run(t_in=trans_rules,y_in=trans_labels,t_in_test=trans_rules_test,
                          y_in_test=trans_labels_test,c_in=class_rules,params=parameters,verbosity=verbosity,logging=logging)
    acc_train <- c(acc_train,accs[1])
    acc_test <- c(acc_test,accs[2])
    train_rules <- c(train_rules,accs[3])
  }
  print(paste('Average baseline accs - train:',mean(baseline_train),'test:',mean(baseline_test)))
  print(paste('    baseline rules:',mean(baseline_train_rules)))
  print(paste('Average cba accs - train:',mean(unlist(cba_train)),'test:',mean(unlist(cba_test))))
  print(paste('    cba rules:',mean(unlist(cba_train_rules))))
  print(paste('Average accs - train:',mean(acc_train),'test:',mean(acc_test)))
  print(paste('    rules:',mean(train_rules)))
}

make_classifier <- function(rules,weights,formula) {
  rule_weights <- rowSums(weights)
  rules_to_keep <- which(rule_weights!=0)
  new_rules <- rules[rules_to_keep]
  new_rule_weights <- rule_weights[rules_to_keep]
  
  classifier <- CBA_ruleset(formula=formula,
      rules=new_rules,
      weights = new_rule_weights,
      method = 'majority'
      )
  return(classifier)
}

run_nested_cross_validation <- function(name,raw_data,class_column,support,confidence,parameters,verbosity,logging,maxlength=10,
                                        compare_rf=T,compare_time=F) {
  num_transactions <- dim(raw_data)[1]
  print(paste('Running cross validation for',name,'data with',num_transactions,'transactions total'))
  folds <- generate_folds(raw_data)
  test_accuracy_list <- c()
  ruleset_size_list <- c()
  models <- list()
  baseline_test_accuracy_list <- c()
  rf_test_accuracy_list <- c()
  for(i in 1:length(folds)) {
    if(i>3) {
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
    test_accuracy_list <- c(test_accuracy_list,model_test_acc)
    ruleset_size_list <- c(ruleset_size_list,best_model$num_rules)
    
    baseline_test_acc <- evaluate_baseline_accuracy(test_trans_rules,test_trans_labels,class_rules,verbosity=verbosity)
    baseline_test_accuracy_list <- c(baseline_test_accuracy_list,baseline_test_acc)
    
    if(compare_rf) {
      rf_test_acc <- evaluate_rf_accuracy(raw_data[train_indices,],raw_data[test_indices,],class_column)
      rf_test_accuracy_list <- c(rf_test_accuracy_list,rf_test_acc)
    }
    if(i==10 & compare_time) {
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
    #cba_test_acc <- evaluate_cba_accuracy()
  }
  print(paste('Average model test accuracy:',mean(test_accuracy_list)))
  print(paste('Average model ruleset size:',mean(ruleset_size_list)))
  print(paste('Average baseline test accuracy:',mean(baseline_test_accuracy_list)))
  print(paste('Average rf test accuracy:',mean(rf_test_accuracy_list)))
}


adam_p <- list(learning_rate=0.1,beta1=0.9,beta2=0.999,epsilon=1e-08)
adadelta_p <- list(learning_rate=0.001,rho=0.95,epsilon=1e-08)
#reg_weights <- list(l1=0.1,l2=0.01)
reg_weights_2 <- list(l1=0.01,l2=0.01)
reg_weights_3 <- list(l1=0.001,l2=0.01)
reg_weights_4 <- list(l1=0.0001,l2=0.01)
reg_weights_5 <- list(l1=0.0001,l2=0.05)
reg_weights_6 <- list(l1=0.001,l2=0.0001)
gridsearch_p <- list(epochs=c(5,10),learning_rate=c(0.05,0.2,0.5),batch_size=16,loss=c('cross'),optimizer=c('adam'),adam_params=adam_p,
          regularization=c('l1'),regularization_weights=list(reg_weights_2,reg_weights_3,reg_weights_4,
                                                             reg_weights_5,reg_weights_6))#,early_stop='test_acc',patience=4L)

run_nested_cross_validation('Breast',prepare_bc(),'V11',0.01,0.5,gridsearch_p,verbosity=0,logging=0,compare_rf = T,compare_time=T)
run_nested_cross_validation('Cleve',prepare_cleve(),'V14',0.01,0.5,gridsearch_p,verbosity=0,logging=1,compare_rf=T,compare_time=T)
run_nested_cross_validation('Glass',prepare_glass(),'V11',0.01,0.5,gridsearch_p,verbosity=0,logging=0,compare_rf=T,compare_time=T)
run_nested_cross_validation('Heart',prepare_heart(),'V14',0.01,0.5,gridsearch_p,verbosity=0,logging=0,compare_rf=T,compare_time=T)
run_nested_cross_validation('Iris',prepare_iris(),'V4',0.01,0.5,gridsearch_p,verbosity=0,logging=0,compare_rf=T,compare_time=T)
#run_nested_cross_validation("Labor",prepare_labor(),'V17',0.01,0.5,gridsearch_p,verbosity=0,logging=0)
run_nested_cross_validation("LED7",prepare_led7(),'V8',0.01,0.5,gridsearch_p,verbosity=0,logging=0,compare_rf=T,compare_time=T)
run_nested_cross_validation("Pima",prepare_pima(),'V9',0.01,0.5,gridsearch_p,verbosity=0,logging=0,compare_rf=T,compare_time=T)
run_nested_cross_validation("Tic",prepare_tic(),'V10',0.01,0.5,gridsearch_p,verbosity=0,logging=0,compare_rf=T,compare_time=T)
#run_nested_cross_validation("Wine",prepare_wine(),'V13',0.01,0.5,gridsearch_p,verbosity=0,logging=0)

#p <- list(epochs=40,learning_rate=0.05,batch_size=16,loss='cross',optimizer='sgd',adam_params=adam_p)
p <- list(epochs=2,learning_rate=0.55,batch_size=16,loss='cross',optimizer='adam',adam_params=adam_p,
          regularization='l1',regularization_weights=reg_weights)#,early_stop='test_acc',patience=4L)
#NOTE: early stopping will increase accuracy at the cost of ruleset size. Also, it's probably fake unless I use a validation set
run_cross_validation('Anneal',prepare_anneal(),'V39',0.01,0.5,p,verbosity=0,logging=1,maxlen=6)
run_cross_validation('Austral',prepare_austral(),'V15',0.01,0.5,p,verbosity=0,logging=1)
#run_cross_validation('Auto',prepare_auto(),'V26',0.04,0.5,p,verbosity=0,logging=1,maxlength=6)#Fails
run_cross_validation('Breast',prepare_bc(),'V11',0.01,0.5,p,verbosity=0,logging=1)
run_cross_validation('CRX',prepare_crx(),'V16',0.01,0.5,p,verbosity=0,logging=0,maxlen=9)
run_cross_validation('Cleve',prepare_cleve(),'V14',0.02,0.5,p,verbosity=0,logging=1)
run_cross_validation('German',prepare_german(),'V21',0.024,0.5,p,verbosity=0,logging=0,maxlen=9)
run_cross_validation('Glass',prepare_glass(),'V11',0.01,0.5,p,verbosity=0,logging=1)
run_cross_validation('Heart',prepare_heart(),'V14',0.01,0.5,p,verbosity=0,logging=1)
run_cross_validation('Hepatic',prepare_hepatic(),'V20',0.03,0.5,p,verbosity=0,logging=0,maxlen=9)#TOO BIG
run_cross_validation('Horse',prepare_horse(),'V40',0.01,0.5,p,verbosity=0,logging=1,maxlen=9)
run_cross_validation("Iono",prepare_iono(),'V35',0.05,0.5,p,verbosity=0,logging=1,maxlen=6)
run_cross_validation("Iris",prepare_iris(),'V5',0.01,0.5,p,verbosity=0,logging=1)
run_cross_validation("Labor",prepare_labor(),'V17',0.03,0.5,p,verbosity=0,logging=1)
run_cross_validation("LED7",prepare_led7(),'V8',0.01,0.5,p,verbosity=0,logging=1)
run_cross_validation("Lymph",prepare_lymph(),'V19',0.04,0.5,p,verbosity=0,logging=1)
run_cross_validation("Pima",prepare_pima(),'V9',0.01,0.5,p,verbosity=0,logging=1)
run_cross_validation("Sick",prepare_sick(),'V30',0.23,0.5,p,verbosity=0,logging=1,maxlen=7)
run_cross_validation("Sonar",prepare_sonar(),'V61',0.04,0.5,p,verbosity=0,logging=1)
run_cross_validation("Tic",prepare_tic(),'V10',0.01,0.5,p,verbosity=0,logging=1)
run_cross_validation("Wine",prepare_wine(),'V13',0.01,0.5,p,verbosity=0,logging=0)
run_cross_validation("Waveform",prepare_waveform(),'V22',0.025,0.5,p,verbosity=0,logging=0,maxlen=9)
run_cross_validation("Vehicle",prepare_vehicle(),'V19',0.05,0.5,p,verbosity=0,logging=0,maxlen=8)
#run_cross_validation("Zoo",prepare_zoo(),'V18',0.055,0.5,p,verbosity=0,logging=1,maxlen=10) #too class-imbalanced
}