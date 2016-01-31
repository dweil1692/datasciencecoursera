#load the necessary libraries
library(ggplot2)
library(gridExtra)
library(caret)
library(randomForest)
library(parallel)
library(doParallel)
#Read in the training/testing datasets
training <- read.csv('training_set.csv', header = TRUE)
testing <- read.csv('testing_set.csv', header = TRUE)
#Remove unnecessary information to shrink down datasets a bit
training <- training[8:length(training)]
testing <- testing[8:length(testing)]
#Remove columns containing a threshold NA% of 80 to remove more unnecessary predictors
training <- training[,colSums(is.na(training)) < nrow(training) * 0.8]
testing <- testing[,colSums(is.na(testing)) < nrow(testing) * 0.8]
#Equalize both datasets to contain the same predictor columns
predictors <- intersect(names(training), names(testing))
#reduced_training <- data.frame(row_id = c(1:nrow(training)))
reduced_training <- data.frame(classe = training$classe)
reduced_testing <- data.frame(row_id = c(1:nrow(testing)))
for(i in 1:length(predictors)) {
  reduced_training = cbind(reduced_training, training[predictors[i]])
  reduced_testing = cbind(reduced_testing, testing[predictors[i]])
}
#reduced_training <- reduced_training[2:length(reduced_training)]
reduced_testing <- reduced_testing[2:length(reduced_testing)]
#Slice training set into training/testing subsets
inTrain <- createDataPartition(reduced_training$classe, p = 0.7, list = FALSE)
ntrain <- reduced_training[inTrain,]
ntest <- reduced_training[-inTrain,]
#Configure model building processes to utilize multi-core parallel processing of data
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
fitControl <- trainControl(method = 'cv', number = 10, allowParallel = TRUE)
#Time to start building some models!
RfModFit <- train(classe ~., data = ntrain, method = 'rf', trControl = fitControl) #Random forest model based using k-fold cross validation
pred <- predict(RfModFit, ntest) #predict outcomes on the test dataset using the above model
#A confusion matrix was constructed to compute/display accuracy statistics
confusionMatrix(ntest$classe, pred)