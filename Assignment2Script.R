#Introduction to Deep Learning
#ASSIGNMENT 2

#-------------------------------------------------------
#Libraries

library(caret)
library(dplyr)
library(magrittr)
library(keras)
library(mlbench)
library(dplyr)
library(neuralnet)


#-------------------------------------------------------
#Task 1: Importing

#load the saved training and test datasets
load(file='Games_training_set.Rda')
load(file='Games_testing_set.Rda')

#combine the two datasets back to one dataframe
df <- rbind(games_training_set2, games_testing_set2)


#-------------------------------------------------------
#Task 2: Data Preparation

#removing unnecessary columns
df = subset(df, select = -c(In.app.Purchases, Age.Rating, Languages, Genres, Original.Release.Date, Current.Version.Release.Date, Size))

#convert all columns of dataframe to numeric except for the ratings
df2 = sapply(df[, 1:18], unclass)
df2 = as.data.frame(df2)
Ratings <- df[, 19]
df = cbind(df2, Ratings)

#one hot encoding the target variable
df %<>% mutate_if(is.character, as.factor)
dummy <- dummyVars(' ~ .', data=df)
newdata <- data.frame(predict(dummy, newdata=df))
df = as.data.frame(newdata)


#-------------------------------------------------------
#Task 3: Build, compile the neural network, and fit training data

#converting dataframe into matrix
matrix <- as.matrix(df)
dimnames(matrix) <- NULL

#setting up training and test datasets
set.seed(5555)
ind <- sample(2, nrow(matrix), replace=T, prob=c(0.8, 0.2))
X_train <- matrix[ind==1, c(1:18)]
X_test <- matrix[ind==2, 1:18]
y_train <- matrix[ind==1, c(19:20)]
y_test <- matrix[ind==2, 19:20]


#MODEL1-------------------------------------------------
#model creation
model <- keras_model_sequential()
model %>%
  layer_dense(units=10, activation='relu', input_shape=c(18)) %>%
  layer_dense(units=5, activation='relu') %>%
  layer_dense(units=2, activation='softmax')
summary(model)

#model back propagation
model %>% compile(loss='mse', optimizer='rmsprop', metrics=c('accuracy'))

#model training
mymodel <- model %>% fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
plot(mymodel)

#model evaluation
model %>% evaluate(X_test, y_test)


#-------------------------------------------------------
#Task 4: Parameter Tuning


#MODEL2-------------------------------------------------
#model2, adding dropout layers
model2 <- keras_model_sequential()
model2 %>% layer_dense(units=10, activation='relu', input_shape=c(18)) %>%
  layer_dropout(rate=0.4) %>%
  layer_dense(units = 5, activation='relu') %>%
  layer_dropout(rate=0.3) %>%
  layer_dense(units=2, activation='softmax')
summary(model2)

#compile
model2 %>% compile(loss='mse', optimizer='rmsprop', metrics=c('accuracy'))

#fit model2
mymodel2 <- model2 %>% fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

#evaluate model2
model2 %>% evaluate(X_test, y_test)


#MODEL3-------------------------------------------------
#model3, lowering dropout layer settings
model3 <- keras_model_sequential()
model3 %>% layer_dense(units=10, activation='relu', input_shape=c(18)) %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units=5, activation='relu') %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units=2, activation='softmax')
summary(model3)

#compile
model3 %>% compile(loss='mse', optimizer='rmsprop', metrics=c('accuracy'))

#fit model3
mymodel3 <- model3 %>% fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

#evaluate model3
model3 %>% evaluate(X_test, y_test)


#MODEL4-------------------------------------------------
#model4, adding another hidden layer
model4 <- keras_model_sequential()
model4 %>% layer_dense(units=10, activation='relu', input_shape=c(18)) %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units = 10, activation='relu') %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units=5, activation='relu') %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units=2, activation='softmax')
summary(model4)

#compile
model4 %>% compile(loss='mse', optimizer='rmsprop', metrics=c('accuracy'))

#fit model4
mymodel4 <- model4 %>% fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

#evaluate model4
model4 %>% evaluate(X_test, y_test)


#MODEL5-------------------------------------------------
#model5, increasing first hidden layer neurons and changing dropout layer rate to decreasing every layer
model5 <- keras_model_sequential()
model5 %>% layer_dense(units=20, activation='relu', input_shape=c(18)) %>%
  layer_dropout(rate=0.4) %>%
  layer_dense(units = 10, activation='relu') %>%
  layer_dropout(rate=0.3) %>%
  layer_dense(units=5, activation='relu') %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units=2, activation='softmax')
summary(model5)

#compile
model5 %>% compile(loss='mse', optimizer='rmsprop', metrics=c('accuracy'))

#fit model5
mymodel5 <- model5 %>% fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

#evaluate model5
model5 %>% evaluate(X_test, y_test)


#MODEL6-------------------------------------------------
#model6, drastically increase the number of neurons for each layer
model6 <- keras_model_sequential()
model6 %>% layer_dense(units=50, activation='relu', input_shape=c(18)) %>%
  layer_dropout(rate=0.4) %>%
  layer_dense(units = 25, activation='relu') %>%
  layer_dropout(rate=0.3) %>%
  layer_dense(units=10, activation='relu') %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units=2, activation='softmax')
summary(model6)

#compile
model6 %>% compile(loss='mse', optimizer='rmsprop', metrics=c('accuracy'))

#fit model6
mymodel6 <- model6 %>% fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

#plot model6
plot(mymodel6)

#evaluate model6
model6 %>% evaluate(X_test, y_test)


#MODEL7-------------------------------------------------
#model7, drastically increasing the number of neurons again
model7 <- keras_model_sequential()
model7 %>% layer_dense(units=100, activation='relu', input_shape=c(18)) %>%
  layer_dropout(rate=0.4) %>%
  layer_dense(units = 50, activation='relu') %>%
  layer_dropout(rate=0.3) %>%
  layer_dense(units=25, activation='relu') %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units=2, activation='softmax')
summary(model7)

#compile
model7 %>% compile(loss='mse', optimizer='rmsprop', metrics=c('accuracy'))

#fit model7
mymodel7 <- model7 %>% fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

#plot model7
plot(mymodel7)

#evaluate model7
model7 %>% evaluate(X_test, y_test)


#MODEL6-------------------------------------------------
#visualizing model6
pred <- predict(model6, X_test)
plot(y_test, pred, xlab = 'Actual', ylab = 'Prediction')
abline(a=0, b=1)