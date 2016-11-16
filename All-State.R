
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

# Please right pad the characters by sign @
encodeCharacter <- function(charcode) {
  r = 0
  ln = nchar(charcode)
  for (i in 1:ln)  
    r = r + (as.integer(charToRaw(substr(charcode,i,i)))-
              as.integer(charToRaw('@')))  * 27^(ln-i)  
  r
}

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

save("x_train","x_test","train_test_orig","y_train_raw",file="All State Processed Data Sets")
rm(train_test)

load("All State Processed Data Sets")
feature_names = names(x_train)

xgb_params = list(
  seed = 0,colsample_bytree = 0.5,colsample_bylevel=1,subsample = 0.9,
  eta = 0.01,max_depth = 12,num_parallel_tree = 1,
  min_child_weight = 1,base_score=7, gamma=1, objective=logregobj
  #objective="reg:linear"
  )

SHIFT = 200
y_train = log(y_train_raw+SHIFT)
#y_train = y_train_raw

###########
#nonOutliers <- which(y_train_raw<50000 & y_train_raw>300)
set.seed(100)
strain <- sample(1:nrow(x_train),0.8*nrow(x_train))
svalid <- setdiff(1:nrow(x_train),strain)



feature.names <- colnames(x_train)
dtrain = xgb.DMatrix(as.matrix(x_train[strain,feature.names,with=FALSE]), label=y_train[strain], missing=NA)
dvalid = xgb.DMatrix(as.matrix(x_train[svalid,feature.names,with=FALSE]), label=y_train[svalid], missing=NA)
#dtest = xgb.DMatrix(as.matrix(x_test), missing=NA)

watchlist <- list(valid=dvalid,train=dtrain)
xgboost.fit <- xgb.train (data=dtrain,xgb_params,missing=NA,early.stop.round = 50,feval=xg_eval_mae,nrounds=10009,
                          maximize=FALSE,verbose=1,watchlist = watchlist)

# importance_matrix <- xgb.importance(model = xgboost.fit, feature.names)
# options(scipen=999)
# print(importance_matrix,200)

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