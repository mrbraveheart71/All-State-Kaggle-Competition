library(data.table)
library(Metrics)
library(xgboost)
library(e1071)  
library(ROCR)
library(gam)
library(mgcv)
library(Metrics)

# Calculates n-way interactions for categorical variables
# If train is true we use folds to get out of sample results
getCategoryNWayInteraction <- function(x_train, y_train, x_test=NULL,trainOrApply = 'T',
                                       category_vector, folds=5, eliminateOrigColumns=TRUE,
                                       simpleName=TRUE) {
  no_rows <- nrow(x_train)
  if (trainOrApply=='A') folds=1
  sample_folds <- sample(1:folds,no_rows,replace=TRUE)
  
  train <- cbind(x_train,y_train)
  test <- x_test
  
  # Set names of columns
  if (simpleName==TRUE) {
    colMeanResponseName <- "NWay_Mean_Response"
    colMedianResponseName <- "NWay_Median_Response"
    colCountResponseName <- "NWay_Count_Response"
    colMaxResponseName <- "NWay_Max_Response"
    colMinResponseName <- "NWay_Min_Response"
    colSDResponseName <- "NWay_SD_Response"
    
  } else {  
    colMeanResponseName <- paste0(paste0(category_vector,collapse="_"),"_Mean_Response")
    colMedianResponseName <- paste0(paste0(category_vector,collapse="_"),"_Median_Response")
    colCountResponseName <- paste0(paste0(category_vector,collapse="_"),"_Count_Response")
    colMaxResponseName <- paste0(paste0(category_vector,collapse="_"),"_Max_Response")
    colMinResponseName <- paste0(paste0(category_vector,collapse="_"),"_Min_Response")
    colSDResponseName <- paste0(paste0(category_vector,collapse="_"),"_SD_Response")
  }
  
  for (f in 1:folds) {
    if (trainOrApply=='T')
      idx_train <- which(!sample_folds==f) else 
        idx_train <- which(sample_folds==f)        
      
      idx_out_of_sample <- which(sample_folds==f) 
      n_Way_Results <- train[idx_train,j=list(Mean.Response=mean(y_train),Median.Response=median(y_train),
                                              Max.Response=max(y_train),Min.Response=min(y_train),SD.Response=sd(y_train),
                                              Count=length(y_train)),by=category_vector]
      setkeyv(n_Way_Results,category_vector)
      
      if (trainOrApply=='T')  {
        train[idx_out_of_sample,colMeanResponseName] <- n_Way_Results[train[idx_out_of_sample,category_vector,with=FALSE], list(Mean.Response)]
        #train[idx_out_of_sample,colMedianResponseName] <- n_Way_Results[train[idx_out_of_sample,category_vector,with=FALSE], list(Median.Response)]
        #train[idx_out_of_sample,colCountResponseName] <- n_Way_Results[train[idx_out_of_sample,category_vector,with=FALSE], list(Count)]
        #train[idx_out_of_sample,colMaxResponseName] <- n_Way_Results[train[idx_out_of_sample,category_vector,with=FALSE], list(Max.Response)]
        #train[idx_out_of_sample,colMinResponseName] <- n_Way_Results[train[idx_out_of_sample,category_vector,with=FALSE], list(Min.Response)]
        #train[idx_out_of_sample,colSDResponseName] <- n_Way_Results[train[idx_out_of_sample,category_vector,with=FALSE], list(SD.Response)]
      } else {
        test[,colMeanResponseName] <- n_Way_Results[test[,category_vector,with=FALSE], list(Mean.Response)]
        #test[,colMedianResponseName] <- n_Way_Results[test[,category_vector,with=FALSE], list(Median.Response)]
        #test[,colCountResponseName] <- n_Way_Results[test[,category_vector,with=FALSE], list(Count)]
        #test[,colMaxResponseName] <- n_Way_Results[test[,category_vector,with=FALSE], list(Max.Response)]
        #test[,colMinResponseName] <- n_Way_Results[test[,category_vector,with=FALSE], list(Min.Response)]
        #test[,colSDResponseName] <- n_Way_Results[test[,category_vector,with=FALSE], list(SD.Response)]
        
      }
  } # end of For Loop with Folds
  
  # returnCols <- c(colMeanResponseName,colMedianResponseName,colCountResponseName, colMaxResponseName,colMinResponseName,colSDResponseName)
  # returnCols <- c(colMeanResponseName, colCountResponseName)
  if (trainOrApply=='T') {
    #return <- train[,returnCols,with=FALSE]
    return <- train
  } else {
    #return <- test[,returnCols,with=FALSE]
    return <- test
  }
  if (eliminateOrigColumns==FALSE) category_vector <- NULL
  returnCols <- setdiff(colnames(return),c(category_vector,"y_train"))
  return[,returnCols,with=FALSE]
}
