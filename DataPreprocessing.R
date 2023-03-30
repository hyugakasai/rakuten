#Introduction to Deep Learning
#Assignment 4

#library(tensorflow)
library(keras)
library(EBImage)
library(stringr)
library(pbapply)

#Set image size
width <- 32
height <- 32

extract_feature <- function(dir_path, width, height, is_low=TRUE, add_label=TRUE) {
  img_size <- width*height
  images_names <- list.files(dir_path)
  if(add_label){
    images_names <- images_names[grepl(ifelse(is_low, "low", "high"), images_names)]
    label <- ifelse(is_low, 0, 1)
  }
  print(paste("Start processing", length(images_names), "images"))
  feature_list <- pblapply(images_names, function(imgname){
    img <- readImage(file.path(dir_path, imgname))
    img_resized <- resize(img, w=width, h=height)
    grayimg <- channel(img_resized, "gray")
    img_matrix <- grayimg@.Data
    img_vector <- as.vector(t(img_matrix))
    return(img_vector)
  })
  feature_matrix <- do.call(rbind, feature_list)
  feature_matrix <- as.data.frame(feature_matrix)
  names(feature_matrix) <- paste0("pixel", c(1:img_size))
  if(add_label){
    feature_matrix <- cbind(label=label, feature_matrix)
  }
  return(feature_matrix)
}

low_data <- extract_feature("low/", width, height)
high_data <- extract_feature("high/", width, height, FALSE)
dim(low_data)
dim(high_data)

save(low_data, file="low.RData")
save(high_data, file="high.RData")
