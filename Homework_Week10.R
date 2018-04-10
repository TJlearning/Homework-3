#Week10_ISLR Chapter 9 Exercises 8
# (a) Create a training set containing a random sample of 800 observations, and a test set containing the remaining observations.
library(ISLR)
?OJ
fix(OJ)
names(OJ)
dim(OJ) ## variable names and dataset dimensions
summary(OJ)

set.seed(42)
train <- sample(nrow(OJ), 800)
OJ.train <- OJ[train, ]
OJ.test <- OJ[-train, ]

# (b) Fit a support vector classifier to the training data using cost=0.01, with Purchase as the response and the other variables as predictors. Use the summary() function to produce summary statistics, and describe the results obtained.
library(e1071)
svm.linear <- svm(Purchase ~ ., data = OJ.train, kernel = "linear", cost = 0.01)
summary(svm.linear)

# (c) What are the training and test error rates?
train.pred <- predict(svm.linear, OJ.train)
table(OJ.train$Purchase, train.pred)
test.pred <- predict(svm.linear, OJ.test)
table(OJ.test$Purchase, test.pred)

# (d) Use the tune() function to select an optimal cost. Consider values in the range 0.01 to 10.
set.seed(42)
tune.out <- tune(svm, Purchase ~ ., data = OJ.train, kernel = "linear", ranges = list(cost = 10^seq(-2, 1, by = 0.25)))
summary(tune.out)

# (e) Compute the training and test error rates using this new value for cost.
svm.linear <- svm(Purchase ~ ., kernel = "linear", data = OJ.train, cost = tune.out$best.parameter$cost)
train.pred <- predict(svm.linear, OJ.train)
table(OJ.train$Purchase, train.pred)

test.pred <- predict(svm.linear, OJ.test)
table(OJ.test$Purchase, test.pred)

# (f) Repeat parts (b) through (e) using a support vector machine with a radial kernel. Use the default value for gamma.


set.seed(42)
tune.out <- tune(svm, Purchase ~ ., data = OJ.train, kernel = "radial", ranges = list(cost = 10^seq(-2, 1, by = 0.25)))
summary(tune.out)

svm.radial <- svm(Purchase ~ ., kernel = "radial", data = OJ.train, cost = tune.out$best.parameter$cost)
summary(svm.radial)
train.pred <- predict(svm.radial, OJ.train)
table(OJ.train$Purchase, train.pred)
test.pred <- predict(svm.radial, OJ.test)
table(OJ.test$Purchase, test.pred)


# (g) Repeat parts (b) through (e) using a support vector machine with a polynomial kernel. Set degree=2.
svm.poly <- svm(Purchase ~ ., kernel = "polynomial", data = OJ.train, degree = 2)
summary(svm.poly)
train.pred <- predict(svm.poly, OJ.train)
table(OJ.train$Purchase, train.pred)
test.pred <- predict(svm.poly, OJ.test)
table(OJ.test$Purchase, test.pred)

set.seed(42)
tune.out <- tune(svm, Purchase ~ ., data = OJ.train, kernel = "polynomial", degree = 2, ranges = list(cost = 10^seq(-2, 1, by = 0.25)))
summary(tune.out)

svm.poly <- svm(Purchase ~ ., kernel = "polynomial", degree = 2, data = OJ.train, cost = tune.out$best.parameter$cost)
summary(svm.poly)

train.pred <- predict(svm.poly, OJ.train)
table(OJ.train$Purchase, train.pred)
test.pred <- predict(svm.poly, OJ.test)
table(OJ.test$Purchase, test.pred)

# (h) Overall, which approach seems to give the best results on this data?
  