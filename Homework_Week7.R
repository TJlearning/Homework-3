#Week7_ ISLR Chapter 8 Exercises 9
library(ISLR)
?OJ
fix(OJ)
names(OJ)
dim(OJ) ## variable names and dataset dimensions
summary(OJ)
# (a) Create a training set containing a random sample of 800 observations, and a test set containing the remaining observations.
library(tree)
set.seed(1)#set seed=2, get different results
train <- sample(1:nrow(OJ), 800)
OJ.train <- OJ[train, ]
OJ.test <- OJ[-train, ]

# (b) Fit a tree to the training data, with Purchase as the response and the other variables as predictors. Use the summary() function to produce summary statistics about the tree, and describe the results obtained. What is the training error rate? How many terminal nodes does the tree have?
tree.oj <- tree(Purchase ~ ., data = OJ.train)
summary(tree.oj)

# (c) Type in the name of the tree object in order to get a detailed text output. Pick one of the terminal nodes, and interpret the information displayed.
tree.oj

# (d) Create a plot of the tree, and interpret the results.
plot(tree.oj)
text(tree.oj, pretty = 0)

# (e) Predict the response on the test data, and produce a confusion matrix comparing the test labels to the predicted test labels. What is the test error rate?
tree.pred <- predict(tree.oj, OJ.test, type = "class")
table(tree.pred, OJ.test$Purchase)

# (f) Apply the cv.tree() function to the training set in order to determine the optimal tree size.
cv.oj <- cv.tree(tree.oj, FUN = prune.misclass)
cv.oj

# (g) Produce a plot with tree size on the x-axis and cross-validated classification error rate on the y-axis.
plot(cv.oj$size, cv.oj$dev, type = "b", xlab = "Tree size", ylab = "Deviance")

# (h) Which tree size corresponds to the lowest cross-validated classification error rate?
prune.oj <- prune.misclass(tree.oj, best = 2)
plot(prune.oj)
text(prune.oj, pretty = 0)

# (i) Produce a pruned tree corresponding to the optimal tree size obtained using cross-validation. If cross-validation does not lead to selection of a pruned tree, then create a pruned tree with five terminal nodes.
summary(tree.oj)
summary(prune.oj)

# (j) Compare the training error rates between the pruned and unpruned trees. Which is higher?
prune.pred <- predict(prune.oj, OJ.test, type = "class")
table(prune.pred, OJ.test$Purchase)

# (k) Compare the test error rates between the pruned and unpruned trees. Which is higher?
  