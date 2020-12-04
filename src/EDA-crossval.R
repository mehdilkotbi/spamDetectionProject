rm(list=ls())
graphics.off()

setwd("/Users/kd/Desktop/R/src")

list.of.packages <- c("xgboost", "ParBayesianOptimization")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(corrplot)
library(caret)
library(xgboost)
library("ParBayesianOptimization")


load("../data/spam_data_train.rda")
head(data_train)

# Correlation analysis
M <-cor(data_train)
corrplot(M, type="upper", order="hclust", tl.cex = 0.5)
findCorrelation(M, cutoff = .75, verbose = TRUE, names = TRUE)


# train/test split
smp_size <- floor(0.7 * nrow(data_train))
set.seed(123)
train_ind <- sample(seq_len(nrow(data_train)), size = smp_size)

train <- data_train[train_ind, ]
test <- data_train[-train_ind, ]

drop <- c('label')

xtrain = as.matrix(train[,!(names(train) %in% drop)])
ytrain = train[,'label']

xtest = as.matrix(test[,!(names(test) %in% drop)])
ytest = test[,'label']

xgbTrain = xgb.DMatrix(data=xtrain,label=ytrain)
xgbTest = xgb.DMatrix(data=xtest,label=ytest)

params = list(
  eta=0.001,
  max_depth=5,
  objective="binary:logistic",
  eval_metric="auc"
)
model = xgb.train(
  params=params,
  data=xgbTrain,
  nrounds=1000,
  nthreads=1,
  early_stopping_rounds=50,
  watchlist=list(val1=xgbTrain,val2=xgbTest),
  verbose=1
)

# Review the final model and results
model

# Create our prediction probabilities
pred <- predict(model, xgbTest)
pred_modf <- ifelse(pred >= 0.52, 1, 0)

confusionMatrix(as.factor(pred_modf), as.factor(ytest), positive='1')

importance_matrix <- xgb.importance(colnames(xtrain), model = model)

xgb.plot.importance(importance_matrix = importance_matrix, top_n = 20, main="Feature importance")




# BO for hyper optimisation

folds <- list(Fold1 = as.integer(seq(1,nrow(xtrain),by = 3)), 
              Fold2 = as.integer(seq(2,nrow(xtrain),by = 3)), 
              Fold3 = as.integer(seq(3,nrow(xtrain),by = 3)))

scoringFunction <- function(eta, max_depth, min_child_weight, subsample) {
  
  Pars <- list( 
    booster = "gbtree"
    , eta = eta
    , max_depth = max_depth
    , min_child_weight = min_child_weight
    , subsample = subsample
    , objective = "binary:logistic"
    , eval_metric = "auc"
  )
  
  xgbcv <- xgb.cv(
    params = Pars
    , data = xgbTrain
    , nround = 5000
    , folds = folds
    , prediction = TRUE
    , showsd = TRUE
    , early_stopping_rounds = 100
    , maximize = TRUE
    , verbose = 0)
  
  return(
    list( 
      Score = max(xgbcv$evaluation_log$test_auc_mean)
      , nrounds = xgbcv$best_iteration
    )
  )
}

bounds <- list( 
  eta = c(0.01, 0.2)
  , max_depth = c(5L, 10L)
  , min_child_weight = c(1, 25)
  , subsample = c(0.25, 1)
)
set.seed(1234)
optObj <- bayesOpt(
  FUN = scoringFunction
  , bounds = bounds
  , initPoints = 6
  , iters.n = 3
)

optObj$scoreSummary
getBestPars(optObj)

