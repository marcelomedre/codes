rm(list = ls())

# Ensemble ML in R

# Load Libraries
libs <- c("mlbench", "caret", "caretEnsemble")
lapply(libs, library, character = TRUE)

# Loading datasets
data("Ionosphere")

data <- Ionosphere
data <- data[,-2] # removing 2nd column because it is a constant
data$V1 <- as.numeric(as.character(data$V1)) # converting factor to num

head(data)

# Bagging. Building multiple models (typically of the same type) from different subsamples of the training dataset.

# CART and random Forest - DEFINING SOME PARAMETERS
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
seed <- 123
metric <- "Accuracy"

# CART
set.seed(seed)
fit.treebag <- train(Class ~., data = data,
                     method = "treebag",
                     metric = metric,
                     trControl = control)

# Random Forest
set.seed(seed)
RF <- train(Class ~., data = data,
            method = "rf",
            metric = metric,
            trControl = control)

# Summary results
bagg_res <- resamples(list(treebag = fit.treebag, rf = RF))
summary(bagg_res)
dotplot(bagg_res)

# -- Stacking Algorithms
# Building multiple models (typically of differing types) and supervisor model
# that learns how to best combine the predictions of the primary models.

# You can combine the predictions of multiple caret models using the caretEnsemble package.

# Example of Stacking Algorithms
# creating submodels

control <- trainControl(method = "repeatedcv",
                        number =  10,
                        repeats = 3, 
                        savePredictions = TRUE,
                        classProbs = TRUE)

# rpart = CART (trees), glm and SVM
algorithmList <- c("rpart", "glm", "svmRadial")
models <- caretList(Class ~., data = data,
                    trControl = control,
                    methodList = algorithmList)
results <- resamples(models)
dotplot(results)
summary(results)

# Correlation between models
modelCor(results)
splom(results)

#stack using glm
stackcontrol <- trainControl(method = "repeatedcv",
                             number = 10,
                             repeats = 3,
                             savePredictions = T,
                             classProbs = T)

set.seed(seed)
stack.glm <- caretStack(models, method = "glm",
                        metric = metric,
                        trControl = stackcontrol)
print(stack.glm)

set.seed(seed)
stack.rf <- caretStack(models, method = "rf", 
                       metric = metric, 
                       trControl = stackcontrol)
print(stack.rf)
