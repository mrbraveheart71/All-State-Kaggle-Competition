
################################################
## R version of most popular local hotels
##################################################
library(data.table)
library(Metrics)
library(xgboost)
library(e1071)  
library(ROCR)
library(gam)
library(mgcv)
library(Metrics)
library(glmnet)
library(stringr)

set.seed(50)
setwd("C:/R-Studio/All-State-Kaggle-Competition")
source("All-State-Functions.R")

# Please right pad the characters by sign @
encodeCharacter <- function(charcode) {
  r = 0
  ln = nchar(charcode)
  for (i in 1:ln)  
    r = r + (as.integer(charToRaw(substr(charcode,i,i)))-
               as.integer(charToRaw('@')))  * 27^(ln-i)  
  r
}

xg_eval_mae <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  #err= mae(y,yhat )
  err= mae(exp(y)-SHIFT,exp(yhat)-SHIFT)
  return (list(metric = "error", value = err))
}

logregobj <- function(preds, dtrain) {
  labels = getinfo(dtrain, "label")
  con = 2
  x = preds - labels
  grad = (con * x) / (abs(x) + con)
  hess = (con^2)  / (abs(x) + con) ^2
  return(list(grad = grad, hess = hess))
}

ID = 'id'
TARGET = 'loss'
SEED = 0

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SUBMISSION_FILE = "sample_submission.csv"

train = fread(TRAIN_FILE, showProgress = TRUE)
test = fread(TEST_FILE, showProgress = TRUE)

y_train_raw = train[,TARGET,with=FALSE][[TARGET]]
#y_train = log(train[,TARGET, with = FALSE]+SHIFT)[[TARGET]]

train[, c(ID, TARGET) := NULL]
test[, c(ID) := NULL]

features = names(train)
orig_cat_names <- colnames(train)[which(colnames(train) %like% 'cat')]
orig_num_names <- colnames(train)[which(colnames(train) %like% 'cont')]

ntrain = nrow(train)
train_test_orig = rbind(train, test)

train_test <- train_test_orig

cat_names_size <- apply(train_test_orig[,orig_cat_names,with=FALSE],2,function(x) max(nchar(x)))

# Pre-processing
# def encode(charcode):
#   r = 0
# ln = len(charcode)
# for i in range(ln):
#   r += (ord(charcode[i])-ord('A')+1)*26**(ln-i-1)
# return r


# Pre-processing
# for (f in features) {
#   if (class(train_test[[f]])=="character") {
#     levels <- unique(train_test[[f]])
#     train_test[[f]] <- as.integer(factor(train_test[[f]], levels=levels))
#   }
# }

for (f in features) {
  print(paste0('Feature : ',f))
  if (class(train_test[[f]])=="character") {
    if (cat_names_size[f] > 1) {
      train_test_orig[[f]] <- apply(train_test_orig[,f,with=FALSE],1, function(x) str_pad(x,cat_names_size[f], side="right", pad="@"))
    }
    train_test[[f]] <- apply(train_test_orig[,f,with=FALSE],1,encodeCharacter)
  }
}

x_train = train_test[1:ntrain,]
x_test = train_test[(ntrain+1):nrow(train_test),]

# for (f in orig_num_names) {
#   print(f)
#   # features_interact <- c("cat103","cat100","cat101","cat102")
#   x_test <- getCategoryNWayInteraction(x_train,y_train,x_test,'A',f,50,eliminateOrigColumns=TRUE,FALSE)
#   x_train <- getCategoryNWayInteraction(x_train,y_train,x_test,'T',f,50,eliminateOrigColumns=TRUE,FALSE)
#   #x_train$new_feature <- log(x_train$cont2)+x_train$cont11+x_train$cont12^-log(train$cont3)+train$cont7^2
# }

save("x_train","x_test","train_test_orig","y_train_raw",
     "orig_cat_names","orig_num_names",
     file="All State Processed Data Sets")
rm(train_test)


# From here load files
load("All State Processed Data Sets")

train_test <- rbind(x_train,x_test)
# Binning
#for (f in orig_num_names) {
for (f in c("cont6")) {
  print(f)
  train_test[[f]]  <- as.integer(factor(cut(train_test[[f]],100)))
}
x_train <- head(train_test,nrow(x_train))
x_test <- tail(train_test,nrow(x_test))

set.seed(100)
strain <- sample(1:nrow(x_train),0.8*nrow(x_train))
svalid <- setdiff(1:nrow(x_train),strain)


xgb_params = list(
  seed = 0,colsample_bytree = 0.5,colsample_bylevel=1,subsample = 0.9,
  eta = 0.05,max_depth = 12,num_parallel_tree = 1,
  min_child_weight = 20,base_score=7, gamma=1, objective=logregobj
  #objective="reg:linear"
)

SHIFT = 200
y_train = log(y_train_raw+SHIFT)

bestScore <- 10000
new_feature_names <- as.vector(NULL)
feature.names <- colnames(x_train)

for (t in 1:10000) {
  feature.names <- colnames(x_train)
  #feature.names <- union(orig_num_names,orig_cat_names)
  dtrain = xgb.DMatrix(as.matrix(x_train[strain,feature.names,with=FALSE]), label=y_train[strain], missing=NA)
  dvalid = xgb.DMatrix(as.matrix(x_train[svalid,feature.names,with=FALSE]), label=y_train[svalid], missing=NA)
  #dtest = xgb.DMatrix(as.matrix(x_test), missing=NA)
  
  watchlist <- list(valid=dvalid,train=dtrain)
  xgboost.fit <- xgb.train (data=dtrain,xgb_params,missing=NA,early.stop.round = 50,feval=xg_eval_mae,nrounds=100,
                            maximize=FALSE,verbose=1,watchlist = watchlist)
  
  importance_matrix <- xgb.importance(model = xgboost.fit, feature.names)
  options(scipen=999)
  print(importance_matrix,200)
  
  xgb.dump(model=xgboost.fit,fname='xgb.dump',fmap='', with.stats=TRUE)
  
  # Did we find another good score
  print(paste0('We found a score of ', xgboost.fit$bestScore,' and the best score so far is : ',bestScore))
  if ((bestScore-0.5) > xgboost.fit$bestScore) {
    print('Found a better score')
    bestScore = xgboost.fit$bestScore
  }  else {
    feature.names <- union(feature.names,delete_feature)
    # x_train <- x_train[,setdiff(feature.names,new_feature_names),with=FALSE]
    # x_test <- x_test[,setdiff(feature.names,new_feature_names),with=FALSE]
  }
  # Try other features
  new_feature_names = as.vector(NULL)
  delete_feature = as.vector(NULL)
  # Create new features
  for (i in 1:1) {
    feature_list <- c("cat1","cat12","cat80","cat81")
    feature_list <- c("cat101","cat57","cat81","cat87")
    feature_list <- c("cat103","cat12","cat80","cat81")
    feature_list <- c("cat1","cat12","cat37","cat38","cat80","cat81")
    
    #feature_list <- sample(intersect(orig_cat_names,importance_matrix$Feature[1:100]),10)
    #delete_feature <- sample(feature.names,1)
    #print(paste0('Delete the feature : ',delete_feature))
    #feature.names <- setdiff(feature.names,delete_feature)
    new_feature_name <- paste(feature_list,collapse="_")
    # print(paste0('Add the ',i,'nth column. New feature names is ',new_feature_name))
    # if ((!new_feature_name %in% new_feature_names) & (!new_feature_name %in% feature.names)) {
    new_feature <- apply(train_test_orig[,feature_list,with=FALSE],1,function(x) encodeCharacter(paste(x,collapse="")))
    x_train[,new_feature_name] <- head(new_feature,nrow(x_train))
    x_test[,new_feature_name] <- tail(new_feature,nrow(x_test))
    #  new_feature_names <- union(new_feature_names, new_feature_name)
    # }
  }
  
}







dtrain = xgb.DMatrix(as.matrix(x_train[,feature.names,with=FALSE]), label=y_train, missing=NA)
dtest = xgb.DMatrix(as.matrix(x_test[,feature.names,with=FALSE]), missing=NA)
# res = xgb.cv(xgb_params,dtrain,nrounds=2000,nfold=4,early_stopping_rounds=15,print_every_n = 10,
#               verbose= 1,feval=xg_eval_mae,maximize=FALSE)
# best_nrounds = res$best_iteration
# cv_mean = res$evaluation_log$test_error_mean[best_nrounds]
# cv_std = res$evaluation_log$test_error_std[best_nrounds]
# cat(paste0('CV-Mean: ',cv_mean,' ', cv_std))

best_nrounds = 2000
watchlist <- list(train=dtrain)
gbdt <- xgb.train (data=dtrain,xgb_params,missing=NA,feval=xg_eval_mae,nrounds=best_nrounds,
                        watchlist=watchlist,  maximize=FALSE,verbose=1)

submission = fread(SUBMISSION_FILE, colClasses = c("integer", "numeric"))
submission$loss = exp(predict(gbdt,dtest))-SHIFT
write.csv(submission,'xgb_starter_v2.sub.csv',row.names = FALSE)

# importance_matrix <- xgb.importance(model = gbdt, feature.names)
# options(scipen=999)
# print(importance_matrix,200)


#1126.19439 on PLB