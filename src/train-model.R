rm(list=ls())
graphics.off()

setwd("/Users/kd/Desktop/R/src")
list.of.packages <- c("xgboost")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(corrplot)
library(caret)
library(xgboost)


load("../data/spam_data_train.rda")
load("../data/spam_data_test.rda")

drop <- c('label')

xtrain = as.matrix(data_train[,!(nrow(data_train) %in% drop)])
ytrain = data_train[,'label']

#xgbTrain = xgb.DMatrix(data=xtrain)
params = list(
  eta=0.01,
  max_depth=6,
  min_child_weight=1,
  subsample=0.5482204,
  objective="binary:logistic",
  eval_metric="auc"
)
model = xgboost(
  params=params,
  data=xtrain,
  label=ytrain,
  nrounds=500,
  nthreads=1,
  verbose=0
)


test = as.matrix(data_test)
pred <- predict(model, test)
write.csv(pred, file = "predictions_test.csv")