#Homework: ISLR has a dataset Khan. It contains gene expression data for 4 types of small round blue cell
#tumors. Use help(Khan) to see details. Apply random forest and boosting to the training set, and tuning the
#hyperparameters to improve the models. Report your main steps and final results.
library(ISLR)
?Khan
#fix(Khan)
summary(Khan)
str(Khan) ## variable names and dataset dimensions
dd = data.frame(Khan$xtrain)
tt = data.frame(Khan$xtest)
str(dd)
summary(dd)
dd$ytrain = as.factor(Khan$ytrain) ## randomForest() requires categorical outcome to be a factor
tt$ytest = as.factor(Khan$ytest)

table(Khan$ytrain)
table(Khan$ytest)

apply(is.na(dd), 2, sum) ## check which variable has missing data
apply(is.na(dd), 1, sum) ## check which observation has missing data

# Random Forests and Bagging
library(randomForest)

set.seed(1)
rf1 = randomForest(ytrain ~ ., data=dd, ntree=1000)
plot(rf1)
plot(rf1$err.rate[,1], type='l', xlab='trees', ylab='Error')
rf1
names(rf1) ## all details are here
rf1$mtry; rf1$ntree ## check what default values were used
rf1$confusion ## same as table(Heart2.train$AHD, rf$predicted)
rf1$err.rate[rf$ntree, ] ## the FPR and FNR can also be obtained here

importance(rf1) ## show rf$importance
varImpPlot(rf1) ## same as dotchart(rf$importance[, 'MeanDecreaseGini']) except order

varImpPlot(randomForest(ytrain ~ ., data=dd)) ## repeat a few times

set.seed(1)
varImpPlot(randomForest(ytrain ~ ., data=dd, mtry=1))
varImpPlot(randomForest(ytrain ~ ., data=dd, mtry=30))
varImpPlot(randomForest(ytrain ~ ., data=dd, mtry=100))

rf2 = randomForest(ytrain ~ ., data=dd, importance=T)
rf2$importance
importance(rf2) ## different from rf$importance except the last column
varImpPlot(rf2)
rf2
plot(rf2)

#Cross-validation to select m
library(caret)
cvCtrl = trainControl(method="repeatedcv", number=5, repeats=4, ## 5-fold CV repeated 4 times
                      #summaryFunction=twoClassSummary,
                      classProbs=TRUE)
set.seed(1)
fitRFcaret = train(x=dd[, 1:2308], y=Khan$ytrain, trControl=cvCtrl,
                   tuneGrid=data.frame(mtry=100:150),
                   #tuneLength=4,
                   #metric="ROC", ## when summaryFunction=twoClassSummary
                   method="rf", ntree=1000) ##  
fitRFcaret
plot(fitRFcaret)

names(fitRFcaret)
fitRFcaret$results
fitRFcaret$bestTune$mtry
fitRFcaret$finalModel
fitRFcaret$finalModel$confusion ## OOB confusion matrix


set.seed(1)
rf3 = randomForest(ytrain ~ ., data=dd, mtry=142, ntree=1000)
rf3
plot(rf3)
varImpPlot(rf3)

#library(rpart)
reprtree:::plot.getTree(rf3, k=30)
reprtree:::plot.getTree(rf3, k=50)

#Boosting
library(survival)
library(lattice)
library(splines)
library(parallel)
library(gbm)
bt1 = gbm(ytrain ~ ., data=dd, distribution="gaussian", n.trees=500)
bt1
names(bt1)
bt1$interaction.depth ## stumps
bt1$cv.folds ## no CV was done

bt4 = gbm(ytrain ~ ., data=dd, distribution="gaussian", n.trees=5000, interaction.depth=4)
mse = function(a,b) mean((a-b)^2)
mse(Khan$ytest, predict(bt1, tt, n.trees=500)) ## MSE= 
mse(Khan$ytest, predict(bt4, tt, n.trees=5000)) ## MSE= 

summary(bt1) ## results and a plot
summary(bt4) ## with d=4, the influence of lstat is smaller
summary(bt1, plotit=F) ## without the plot
sum(summary(bt1, plotit=F)$rel.inf) ## 100
sum(summary(bt4, plotit=F)$rel.inf) ## 100

bt.try = gbm(ytrain ~ ., data=dd, distribution="gaussian", n.trees=100, bag.fraction=1)
summary(bt.try, plotit=F)$rel.inf

par(mfrow=c(2,2))
plot(bt4, i="X1194", main='bt4')
plot(bt4, i="X1003", main='bt4')
plot(bt1, i="X1194", main='bt1')
plot(bt1, i="X1003", main='bt1')

#a 2-dimensional partial dependence plot.
plot(bt4, i=c("X1194", "X1003"))

### look at model performance at the end of 1000, 2000, etc. trees.
set.seed(1)
bt4b = gbm(ytrain ~ ., data=dd, distribution="gaussian", n.trees=2000, interaction.depth=4,shrinkage =0.01)
mse(Khan$ytest, predict(bt4b, tt, n.trees=2000))
bt4c = gbm(ytrain ~ ., data=dd, distribution="gaussian", n.trees=5000, interaction.depth=4,shrinkage =0.01)
mse(Khan$ytest, predict(bt4c, tt, n.trees=5000))
bt4d = gbm(ytrain ~ ., data=dd, distribution="gaussian", n.trees=5000, interaction.depth=4,shrinkage =0.1)
mse(Khan$ytest, predict(bt4d, tt, n.trees=5000))

#Cross-validation using caret
library(caret)
set.seed(1)
ctr = trainControl(method="cv", number=3) ## 3-fold CV
mygrid = expand.grid(n.trees=seq(50, 1000, 50), interaction.depth=1:8,
                     shrinkage=0.01, n.minobsinnode=5)
boost.caretk <- train(ytrain ~ ., dd, trControl=ctr, method='gbm',
                     tuneGrid=mygrid,
                     preProc=c('center','scale'), verbose=F)
boost.caretk
plot(boost.caretk)

#Using the optimal hyperparameters selected by train() improves the result!
boost.caretk$bestTune
mse(Khan$ytest, predict(boost.caretk, tt)) ## MSE=10.8
names(boost.caretk)
boost.caretk$results
boost.caretk$finalModel
