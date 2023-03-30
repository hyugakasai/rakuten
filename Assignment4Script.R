#Introduction to Deep Learning
#Assignment 4

library(keras)
library(EBImage)
library(stringr)
library(pbapply)
library(caret)

load("low.RData")
load("high.RData")

complete_set <- rbind(low_data, high_data)
training_index <- createDataPartition(complete_set$label, p=.8, times=1)
training_index <- unlist(training_index)
train_set <- complete_set[training_index,]
dim(train_set)
test_set <- complete_set[-training_index,]
dim(test_set)

train_data <- data.matrix(train_set)
train_x <- t(train_data[,-1])
train_y <- train_data[,1]
train_array <- train_x
dim(train_array) <- c(ncol(train_x),32, 32, 1)
dim(train_array)

test_data <- data.matrix(test_set)
test_x <- t(test_set[,-1])
test_y <- test_set[,1]
test_array <- test_x
dim(test_array) <- c(ncol(test_x),32, 32, 1)
dim(test_array)


model <- keras_model_sequential() %>%
  layer_conv_2d(filters=32, kernel_size=c(3,3), activation="relu",
                input_shape=c(32,32,1)) %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_conv_2d(filters=64, kernel_size=c(3,3), activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_conv_2d(filters=64, kernel_size=c(3,3), activation="relu") %>%
  layer_dropout(rate=0.3) %>%
  layer_flatten() %>%
  layer_dense(units=32, activation="relu") %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units=1, activation="sigmoid")

summary(model)

model %>% compile(
  loss="binary_crossentropy",
  optimizer="adam",
  metrics=c("accuracy")
)

history <- model %>%
  fit(
    x=train_array, y=train_y,
    epochs=100, batch_size=100,
    validation_split=0.2,
    verbose=2
  )

plot(history)
evaluate(model, test_array, test_y, verbose=2)
