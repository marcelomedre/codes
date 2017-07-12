# R Machine Learning in Parallel with GLM, SVM and AdaBoost

# Setup Parallel Processing

# number of bootstrap samples to create
rm(list = ls())
sampleCount <- 8
library(doSNOW)

cluster <- makeCluster(sampleCount) # # of processors / hyperthreads of machine
registerDoSNOW(cluster)

# Create sample Test and Train Data

url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
magicGamma <- read.table(url, header = FALSE, sep = ",")

# create uniqu id for magicGamma file
magicGamma <- data.frame(id = 1:nrow(magicGamma), magicGamma)

# Spliting into test and train sets
library(caret)
inTrain <- createDataPartition(magicGamma$id, p = 0.8, list = FALSE)
trainData <- magicGamma[inTrain,]
testData <- magicGamma[-inTrain,]

# Randomly Boostraping training samples
trainSamples <- foreach(i = 1:sampleCount) %dopar% {
  trainData[sample(1:nrow(trainData),
                   size = 0.2*nrow(trainData),
                   replace = TRUE),]
}

# Create Function to measure accuracy

accuracy <- function(truth, predicted){
  tTable <- table(truth, predicted)
  print(tTable)
  tp <- tTable[1,1]
  if(ncol(tTable) > 1){fp <- tTable[1,2]} else {fp <- 0}
  if(nrow(tTable) > 1){fn <- tTable[2,1]} else {fn <- 0}
  if(ncol(tTable) > 1 & nrow(tTable) > 1){tn <- tTable[2,2]} else {tn <- 0}
  
  return((tp + tn) / (tp + tn + fp + fn))
}

# Glm all

timer <- proc.time()
glmAll <- glm(V11 ~., trainData[,-1], family = binomial())
proc.time() - timer

timer <- proc.time()
glmAlltest <- predict(glmAll, testData[,c(-1, -ncol(testData))])
proc.time() - timer

# add predicted class and actual class to test data
glmAllResults <- data.frame(id = testData$id, actualClass = testData$V11)
glmAllResults$predictedClass <- ifelse(glmAlltest < 0, "g", "h")

# calculate glm all model accuracy
accuracy(glmAllResults$actualClass, glmAllResults$predictedClass)


# SVM all
library(e1071)
timer <- proc.time()
svmAll <- svm(V11 ~., trainData[,-1],
              kernel = "radial",
              probability = TRUE,
              cost = 10,
              gamma = 0.1)
proc.time() - timer

system.time(svmAllTest <- predict(svmAll, testData[,c(-1,-ncol(testData))],
                                  probability = TRUE)
              )

# add predicted class and actual class to test data
svmAllResults <- data.frame(id = testData$id, actualClass = testData$V11, 
                            predicted = svmAllTest)

# calculate glm all model accuracy
accuracy(svmAllResults$actualClass, svmAllResults$predicted)

# Create different bootstrap Models in Parallel
# glm
system.time(
  modelDataglm <- foreach(i = 1:sampleCount) %dopar% {
    glm(V11 ~., trainSamples[[i]][,-1], family = binomial())
  }
)

# svm
# Could run tune.svm() in parallelbefore this step if needed to get the best
#   values for the cost and gamma parameters.... (very slow)

system.time(
  modelDatasvm <- foreach(i = 1:sampleCount) %dopar% {
    library(e1071)
    svm(V11 ~., trainSamples[[i]][,-1],
        kernel = "radial",
        probability = TRUE,
        cost = 10,
        gamma = 0.1)
  }
)

# Predicting Test Data in Parallel
# glm
system.time(
  predictDataglm <- foreach(i = 1:sampleCount) %dopar% {
    predict(modelDataglm[[i]], testData[,c(-1,-ncol(testData))])
  }
)

# svm
system.time(
  predictDatasvm <- foreach(i = 1:sampleCount) %dopar% {
    library(e1071)
    predict(modelDatasvm[[i]], testData[,c(-1, - ncol(testData))],
            probability = TRUE)
  }
)

