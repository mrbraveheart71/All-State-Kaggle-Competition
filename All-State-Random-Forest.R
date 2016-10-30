################################################
##################################################
library(data.table)
library(Metrics)
library(xgboost)
library(e1071)  
library(ROCR)
library(gam)
library(mgcv)
library(Metrics)
library(randomForest)

set.seed(50)
setwd("C:/R-Studio/All-State-Kaggle-Competition")
source("All-State-Functions.R")

ID = 'id'
TARGET = 'loss'
SEED = 0

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SUBMISSION_FILE = "sample_submission.csv"

train = fread(TRAIN_FILE, showProgress = TRUE)
test = fread(TEST_FILE, showProgress = TRUE)

y_train = log(train[,TARGET, with = FALSE])[[TARGET]]

train[, c(ID, TARGET) := NULL]
test[, c(ID) := NULL]

ntrain = nrow(train)
train_test = rbind(train, test)

features = names(train)

# Pre-processing
for (f in features) {
  if (class(train_test[[f]])=="character") {
    levels <- unique(train_test[[f]])
    train_test[[f]] <- as.integer(factor(train_test[[f]], levels=levels))
  }
}

x_train = train_test[1:ntrain,]
x_test = train_test[(ntrain+1):nrow(train_test),]

feature_names <- colnames(x_train)
category_names <- colnames(x_train)[which(colnames(x_train) %like% 'cat')]

for (cat in category_names) {
  print(paste0('Work on category ',cat))
  x_test <- getCategoryNWayInteraction(x_train,y_train,x_test,'A',cat,5,eliminateOrigColumns=TRUE,FALSE)
  x_train <- getCategoryNWayInteraction(x_train,y_train,x_test,'T',cat,5,eliminateOrigColumns=TRUE,FALSE)
}

mean_y <- mean(y_train)

# replace na with mean
for (i in names(x_train))
  x_train[is.na(get(i)),i:=mean_y,with=FALSE]
for (i in names(x_test))
  x_test[is.na(get(i)),i:=mean_y,with=FALSE]

feature.names <- colnames(x_train)

s <- sample(1:nrow(x_train),0.2*nrow(x_train))
rf.fit <- randomForest(x_train[s],y=y_train[s],ntree=100,mtry=2,do.trace=TRUE)
mae(exp(y_train[s]),exp(rf.fit$predicted))
varImpPlot(rf.fit)




