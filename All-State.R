
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

set.seed(50)
setwd("C:/R-Studio/AllState")

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

ntrain = nrow(train)
train_test = rbind(train, test)

# Pre-processing
encodeCharacter <- function(charcode) {
  r = 0
  ln = nchar(charcode)
  if (ln > 2) {
    print("Error: Expected Maximum of Two Characters!")
    exit
  }
  firstLetter <- as.integer(charToRaw(substring(charcode,1,1)))-as.integer(charToRaw('A'))
  secondLetter <- 0
  if (ln > 1) secondLetter <- as.integer(charToRaw(substring(charcode,2,2)))-as.integer(charToRaw('A'))+1
  r = firstLetter*27+secondLetter
  r
}

# Pre-processing
for (f in features) {
  if (class(train_test[[f]])=="character") {
    levels <- unique(train_test[[f]])
    train_test[[f]] <- as.integer(factor(train_test[[f]], levels=levels))
  }
}

# for (f in features) {
#   print(paste0('Feature : ',f))
#   if (class(train_test[[f]])=="character") {
#     train_test[[f]] <- apply(train_test[,f,with=FALSE],1,encodeCharacter)
#   }
# }
# 
x_train = train_test[1:ntrain,]
x_test = train_test[(ntrain+1):nrow(train_test),]

#save("x_train","x_test",file="All State Processed Data Sets")

xgb_params = list(
  seed = 0,colsample_bytree = 0.5,colsample_bylevel=1,subsample = 0.9,
  eta = 0.005,max_depth = 12,num_parallel_tree = 1,
  min_child_weight = 1,base_score=7, gamma=1, #objective=ln_cosh_obj
  objective="reg:linear"
  )

SHIFT = 180
y_train = log(y_train_raw+SHIFT)
#y_train = y_train_raw

xg_eval_mae <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  #err= mae(y,yhat )
  err= mae(exp(y)-SHIFT,exp(yhat)-SHIFT)
  return (list(metric = "error", value = err))
}

s <- sample(1:nrow(x_train),0.8*nrow(x_train))
feature.names <- colnames(x_train)
dtrain = xgb.DMatrix(as.matrix(x_train[s,feature.names,with=FALSE]), label=y_train[s], missing=NA)
dvalid = xgb.DMatrix(as.matrix(x_train[-s,feature.names,with=FALSE]), label=y_train[-s], missing=NA)
#dtest = xgb.DMatrix(as.matrix(x_test), missing=NA)

set.seed(100)
watchlist <- list(valid=dvalid,train=dtrain)
xgboost.fit <- xgb.train (data=dtrain,xgb_params,missing=NA,early.stop.round = 10,feval=xg_eval_mae,nrounds=10009,
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