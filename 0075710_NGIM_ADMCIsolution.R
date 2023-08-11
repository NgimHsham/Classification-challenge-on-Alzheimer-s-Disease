
# Read the csv file using read.csv function
ADMCItrain <- read.csv("ADMCItrain.csv")
ADMCItrain$Label <- ifelse(ADMCItrain$Label == "AD", 1, 0)

ncol(ADMCItrain)
# convert a binary categorical variable into a numeric variable
head(ADMCItrain[, c(1:5, ncol(ADMCItrain))], 27)
#check the number of negative VS. positive samples to check if we need to handle class imbalance
table(ADMCItrain$Label)
# Removing the ID column from the features
ADMCItrain_df <- ADMCItrain
ADMCItrain_df$ID <- NULL
colnames(ADMCItrain_df)

###########################################################
# Calculate correlation
##########################################################
correlations <- cor(ADMCItrain[, -c(1, ncol(ADMCItrain))], ADMCItrain$Label)

# Create a data frame from correlations
correlation_df <- as.data.frame(correlations)

# Add names as a column
correlation_df$Feature <- row.names(correlation_df)

# Rename the first column to 'Correlation'
colnames(correlation_df)[1] <- "Correlation"

# Get absolute value of correlations
correlation_df$Correlation <- abs(correlation_df$Correlation)

# Sort the data frame by absolute value of Correlation
correlation_df <- correlation_df[order(-correlation_df$Correlation),]

correlation_df

# Filter the features with absolute correlation > 0.5
important_features <- correlation_df[correlation_df$Correlation > 0.30, "Feature"]

print(important_features)
length(important_features)

# Add the target variable "Label" to the vector
important_features <- c(important_features, "Label")

# Subset the data to keep only these features
ADMCItrain_selected <- ADMCItrain[ , colnames(ADMCItrain) %in% important_features]

# Check the dimensions of the new dataframe
head(ADMCItrain_selected[, c(1:5, ncol(ADMCItrain_selected))], 5)

print(ncol(ADMCItrain_selected))

###########################################################
# Visualization
###########################################################
library(ggplot2)
library(corrplot)

# Calculate the correlation matrix
cor_matrix <- cor(ADMCItrain_selected[, -ncol(ADMCItrain_selected)])

# Set the resolution and size of the output graphics device
png("ADMCI_correlation_matrix.png", width = 30, height = 30, units = "in", res = 500)

# Create a full correlation matrix plot
corrplot(cor_matrix, type = "full", tl.col = "black", tl.srt = 45,
         method = "color", col = colorRampPalette(c("#F7FBFF", "#4292C6", "#08306B"))(100),
         addCoef.col = "black", tl.cex = 0.8, cl.cex = 0.8, number.cex = 0.8,
         diag = FALSE, outline = TRUE, mar = c(0, 0, 1, 0),
         cl.pos = "n", cl.length = 0, number.color = "white", number.digits = 2,
         addgrid = TRUE)

# Close the graphics device
dev.off()

# Set the resolution
dpi <- 300

# Open a new png device
png("ADMCI_dendrogram.png", width = 28 * dpi, height = 6 * dpi, res = dpi)

# Perform hierarchical clustering
hc <- hclust(as.dist(1 - abs(cor_matrix)))

# Plot the dendrogram of the hierarchical clustering
plot(hc, main = "Dendrogram of Hierarchical Clustering")

# Close the png device
dev.off()


###########################################################
# Train the model only on the reduced number of features:
###########################################################
# Split data into training and validation sets
validation_ratio <- 0.25
validation_index <- createDataPartition(ADMCItrain_selected$Label, p = validation_ratio, list = FALSE)
training_data <- ADMCItrain_selected[-validation_index, ]
validation_data <- ADMCItrain_selected[validation_index, ]
training_data$Label <- ifelse(training_data$Label == 1, 'AD', 'MCI')
validation_data$Label <- ifelse(validation_data$Label == 1, 'AD', 'MCI')
validation_data$Label

###########################################################
# Recursive Feature Elimination (RFE):
###########################################################
# Load necessary libraries
library(caret)

# Define function to perform RFE
rfe_model <- function(train_data, method, ...) {
  # Train the model
  model <- train(Label ~ ., data = train_data, method = method, trControl = trainControl(method = "cv", number = 10, classProbs = TRUE), metric = "ROC", ...)
  cat("Model training completed. Starting Recursive Feature Elimination...\n")
  # Recursive Feature Elimination
  control <- rfeControl(functions=rfFuncs, method="cv", number=10)
  results <- rfe(train_data[, -ncol(train_data)], train_data$Label, sizes = 1:(ncol(train_data) - 1), rfeControl = control)
  cat("Recursive Feature Elimination completed. Printing top 5 variables...\n")
  # Print the top 5 variables
  cat(paste0("\nTop 5 variables for ", method, " model:\n"))
  print(results$optVariables[1:5])
  cat("Function execution completed.\n")
  
  # Return model and RFE results
  return(list("model" = model, "rfe" = results))
}

# Now call the function for each model type
training_data$Label <- as.factor(ifelse(training_data$Label == 'AD', 'AD', 'MCI'))
validation_data$Label <- as.factor(ifelse(validation_data$Label == 'AD', 'AD', 'MCI'))
validation_data$Label
svm_model <- rfe_model(training_data, "svmLinear", preProcess = c("center", "scale"))
rf_model <- rfe_model(training_data, "rf")
logistic_model <- rfe_model(training_data, "glm", family = "binomial")
knn_model <- rfe_model(training_data, "kknn")
gbm_model <- rfe_model(training_data, "gbm", verbose = FALSE)
xgb_model <- rfe_model(training_data, "xgbTree")
nb_model <- rfe_model(training_data, "naive_bayes")
dt_model <- rfe_model(training_data, "rpart")
nn_model <- rfe_model(training_data, "nnet")

#############################################
# Machine Learning models
#############################################
# Load necessary libraries
library(caret)
library(e1071)
library(randomForest)
library(pROC)
library(mltools)
library(kknn)
library(gbm)
library(xgboost)
library(e1071)
library(rpart)
library(nnet)
set.seed(123)

# Create a vector to store the evaluation metrics
eval_metrics <- data.frame(Model = character(), AUC = numeric(), MCC = numeric(), stringsAsFactors = FALSE)

# Define the cross-validation control
cv <- trainControl(method = "cv", number = 5, classProbs = TRUE)

# Define function to train and evaluate a model
train_model <- function(rfe_model, all_train_data, all_val_data, method, file_path, prob = FALSE, ...) {
  # Select the best features chosen by RFE method
  optimal_vars <- rfe_model$rfe$optVariables
  train_data <- all_train_data[, c(optimal_vars, "Label")]
  val_data <- all_val_data[, c(optimal_vars, "Label")]
  print(ncol(train_data))
  print(ncol(val_data))
  
  # Train the model
  model <- train(Label ~ ., data = train_data, method = method, trControl = cv, metric = "ROC", ...)
  # Save the model to a file
  saveRDS(model, file_path)
  # Load the model from a file
  model <- readRDS(file_path)
  # Predict using the model
  if (prob) {
    predictions_prob <- predict(model, newdata = val_data, type = "prob")
    predictions <- ifelse(predictions_prob[, "AD"] > 0.5, 1, 0)
  } else {
    predictions <- predict(model, newdata = val_data)
    predictions <- as.factor(ifelse(predictions == "AD", 1, 0))
  }
  
  # Prepare for ROC and MCC calculations
  val_data$Label <- as.numeric(ifelse(val_data$Label == 'AD', 1, 0))
  predictions <- as.numeric(ifelse(predictions == 1, 1, 0))
  
  # Calculate AUC and MCC
  auc <- roc(val_data$Label, as.numeric(as.character(predictions)))$auc
  mcc <- mcc(preds = as.numeric(as.character(predictions)), actuals = val_data$Label)
  
  # Append to evaluation metrics
  eval_metrics <<- rbind(eval_metrics, c(method, auc, mcc))
  
  # Print evaluation metrics
  print(eval_metrics)
  
  # Return model
  return(model)
}

# Training all the models on the newly selected features
svm_m <- train_model(svm_model, training_data, validation_data, "svmLinear", "./ADMCI/svm_model.rds", preProcess = c("center", "scale"))
svmRadial_m <- train_model(svm_model, training_data, validation_data, "svmRadial", "./ADMCI/svmRadial_model.rds", preProcess = c("center", "scale"))
svmPoly_m <- train_model(svm_model, training_data, validation_data, "svmPoly", "./ADMCI/svmPoly_model.rds", preProcess = c("center", "scale"))
svmLinearWeights_m <- train_model(svm_model, training_data, validation_data, "svmLinearWeights", "./ADMCI/svmLinearWeights_model.rds", preProcess = c("center", "scale"))
svmRadialWeights_m <- train_model(svm_model, training_data, validation_data, "svmRadialWeights", "./ADMCI/svmRadialWeights_model.rds", preProcess = c("center", "scale"))
rf_m <- train_model(rf_model, training_data, validation_data, "rf", "./ADMCI/rf_model.rds", preProcess = c("center", "scale"))
lr_m <- train_model(logistic_model, training_data, validation_data, "glm", "./ADMCI/lr_model.rds", prob = TRUE, family = "binomial", preProcess = c("center", "scale"))
knn_m <- train_model(knn_model, training_data, validation_data, "kknn", "./ADMCI/knn_model.rds", preProcess = c("center", "scale"))
gbm_m <- train_model(gbm_model, training_data, validation_data, "gbm", "./ADMCI/gbm_model.rds", verbose = FALSE, preProcess = c("center", "scale"))
xgb_m <- train_model(xgb_model, training_data, validation_data, "xgbTree", "./ADMCI/xgb_model.rds", preProcess = c("center", "scale"))
nb_m <- train_model(nb_model, training_data, validation_data, "naive_bayes", "./ADMCI/nb_model.rds", prob = TRUE, preProcess = c("center", "scale"))
dt_m <- train_model(dt_model, training_data, validation_data, "rpart", "./ADMCI/dt_model.rds", prob = TRUE, preProcess = c("center", "scale"))
nn_m <- train_model(nn_model, training_data, validation_data, "nnet", "./ADMCI/nn_model.rds", prob = TRUE, preProcess = c("center", "scale"))

# Rename column names
colnames(eval_metrics) <- c("Model", "AUC", "MCC")
# Print the evaluation metrics for each model
print(eval_metrics)
# Select the best model based on AUC
best_model <- eval_metrics[which.max(eval_metrics$AUC), "Model"]
print(paste("Best Model:", best_model))
#########################################
# Train with PCA 
#########################################
# Create a vector to store the evaluation metrics
eval_metrics_pca <- data.frame(Model = character(), AUC = numeric(), MCC = numeric(), stringsAsFactors = FALSE)

# Define the cross-validation control
cv <- trainControl(method = "cv", number = 5, classProbs = TRUE)

# Define function to train and evaluate a model
train_pca_model <- function(rfe_model, all_train_data, all_val_data, method, file_path, prob = FALSE, ...) {
  train_data <- all_train_data
  val_data <- all_val_data
  
  # Scale the data before PCA
  train_data_scaled <- scale(train_data[ , -ncol(train_data)])  # Exclude the Label column
  val_data_scaled <- scale(val_data[ , -ncol(val_data)])  # Exclude the Label column
  
  # Apply PCA
  pca_train <- prcomp(train_data_scaled)
  pca_val <- predict(pca_train, val_data_scaled)  # Apply same rotation and scaling from train to val data
  
  # Using only the first n components and Add the Label column back
  n <- 10
  pca_train_data <- data.frame(pca_train$x[, 1:n], Label = train_data$Label)
  pca_val_data <- data.frame(pca_val[, 1:n], Label = val_data$Label)
  
  
  # Train the model
  model <- train(Label ~ ., data = pca_train_data, method = method, trControl = cv, metric = "ROC", ...)
  
  # Save the model to a file
  saveRDS(model, file_path)
  
  # Load the model from a file
  model <- readRDS(file_path)
  
  # Predict using the model
  if (prob) {
    predictions_prob <- predict(model, newdata = pca_val_data, type = "prob")
    predictions <- ifelse(predictions_prob[, "AD"] > 0.5, 1, 0)
  } else {
    predictions <- predict(model, newdata = pca_val_data)
    predictions <- as.factor(ifelse(predictions == "AD", 1, 0))
  }
  
  # Prepare for ROC and MCC calculations
  pca_val_data$Label <- as.numeric(ifelse(pca_val_data$Label == 'AD', 1, 0))
  predictions <- as.numeric(ifelse(predictions == 1, 1, 0))
  
  # Calculate AUC and MCC
  auc <- roc(pca_val_data$Label, as.numeric(as.character(predictions)))$auc
  mcc <- mcc(preds = as.numeric(as.character(predictions)), actuals = pca_val_data$Label)
  
  # Append to evaluation metrics
  eval_metrics_pca <<- rbind(eval_metrics_pca, c(method, auc, mcc))
  
  # Print evaluation metrics
  print(eval_metrics_pca)
  
  # Return model
  return(model)
}

# Training all the models on the newly selected features
svm_pca_m <- train_pca_model(svm_model, training_data, validation_data, "svmLinear", "./ADMCI/svm_pca_model.rds", preProcess = c("center", "scale"))
svmRadial_pca_m <- train_pca_model(svm_model, training_data, validation_data, "svmRadial", "./ADMCI/svmRadial_pca_model.rds", preProcess = c("center", "scale"))
svmPoly_pca_m <- train_pca_model(svm_model, training_data, validation_data, "svmPoly", "./ADMCI/svmPoly_pca_model.rds", preProcess = c("center", "scale"))
svmLinearWeights_pca_m <- train_pca_model(svm_model, training_data, validation_data, "svmLinearWeights", "./ADMCI/svmLinearWeights_pca_model.rds", preProcess = c("center", "scale"))
svmRadialWeights_pca_m <- train_pca_model(svm_model, training_data, validation_data, "svmRadialWeights", "./ADMCI/svmRadialWeights_pca_model.rds", preProcess = c("center", "scale"))
rf_pca_m <- train_pca_model(rf_model, training_data, validation_data, "rf", "./ADMCI/rf_pca_model.rds", preProcess = c("center", "scale"))
lr_pca_m <- train_pca_model(logistic_model, training_data, validation_data, "glm", "./ADMCI/lr_pca_model.rds", prob = TRUE, family = "binomial", preProcess = c("center", "scale"))
knn_pca_m <- train_pca_model(knn_model, training_data, validation_data, "kknn", "./ADMCI/knn_pca_model.rds", preProcess = c("center", "scale"))
gbm_pca_m <- train_pca_model(gbm_model, training_data, validation_data, "gbm", "./ADMCI/gbm_pca_model.rds", verbose = FALSE, preProcess = c("center", "scale"))
xgb_pca_m <- train_pca_model(xgb_model, training_data, validation_data, "xgbTree", "./ADMCI/xgb_pca_model.rds", preProcess = c("center", "scale"))
nb_pca_m <- train_pca_model(nb_model, training_data, validation_data, "naive_bayes", "./ADMCI/nb_pca_model.rds", prob = TRUE, preProcess = c("center", "scale"))
dt_pca_m <- train_pca_model(dt_model, training_data, validation_data, "rpart", "./ADMCI/dt_pca_model.rds", prob = TRUE, preProcess = c("center", "scale"))
nn_pca_m <- train_pca_model(nn_model, training_data, validation_data, "nnet", "./ADMCI/nn_pca_model.rds", prob = TRUE, preProcess = c("center", "scale"))
# Rename column names
colnames(eval_metrics_pca) <- c("Model", "AUC", "MCC")
# Print the evaluation metrics for each model
print(eval_metrics_pca)
# Select the best model based on AUC
best_model <- eval_metrics_pca[which.max(eval_metrics_pca$AUC), "Model"]
print(paste("Best Model:", best_model))

#########################################
# Train with RFE and PCA 
#########################################
# Create a vector to store the evaluation metrics
eval_metrics_rfe_pca <- data.frame(Model = character(), AUC = numeric(), MCC = numeric(), stringsAsFactors = FALSE)

# Define the cross-validation control
cv <- trainControl(method = "cv", number = 5, classProbs = TRUE)

# Define function to train and evaluate a model
train_rfe_pca_model <- function(rfe_model, all_train_data, all_val_data, method, file_path, prob = FALSE, ...) {
  # Select the best features chosen by RFE method
  optimal_vars <- rfe_model$rfe$optVariables
  train_data <- all_train_data[, c(optimal_vars, "Label")]
  val_data <- all_val_data[, c(optimal_vars, "Label")]
  
  # Scale the data before PCA
  train_data_scaled <- scale(train_data[ , -ncol(train_data)])  # Exclude the Label column
  val_data_scaled <- scale(val_data[ , -ncol(val_data)])  # Exclude the Label column
  
  # Apply PCA
  pca_train <- prcomp(train_data_scaled)
  pca_val <- predict(pca_train, val_data_scaled)  # Apply same rotation and scaling from train to val data
  
  # Add the Label column back
  pca_train_data <- data.frame(pca_train$x, Label = train_data$Label)
  pca_val_data <- data.frame(pca_val, Label = val_data$Label)
  
  # Train the model
  model <- train(Label ~ ., data = pca_train_data, method = method, trControl = cv, metric = "ROC", ...)
  
  # Save the model to a file
  saveRDS(model, file_path)
  
  # Load the model from a file
  model <- readRDS(file_path)
  
  # Predict using the model
  if (prob) {
    predictions_prob <- predict(model, newdata = pca_val_data, type = "prob")
    predictions <- ifelse(predictions_prob[, "AD"] > 0.5, 1, 0)
  } else {
    predictions <- predict(model, newdata = pca_val_data)
    predictions <- as.factor(ifelse(predictions == "AD", 1, 0))
  }
  
  # Prepare for ROC and MCC calculations
  pca_val_data$Label <- as.numeric(ifelse(pca_val_data$Label == 'AD', 1, 0))
  predictions <- as.numeric(ifelse(predictions == 1, 1, 0))
  
  # Calculate AUC and MCC
  auc <- roc(pca_val_data$Label, as.numeric(as.character(predictions)))$auc
  mcc <- mcc(preds = as.numeric(as.character(predictions)), actuals = pca_val_data$Label)
  
  # Append to evaluation metrics
  eval_metrics_rfe_pca <<- rbind(eval_metrics_rfe_pca, c(method, auc, mcc))
  
  # Print evaluation metrics
  print(eval_metrics_rfe_pca)
  
  # Return model
  return(model)
}

# Training all the models on the newly selected features
svm_rfe_pca_m <- train_rfe_pca_model(svm_model, training_data, validation_data, "svmLinear", "./ADMCI/svm_model.rds", preProcess = c("center", "scale"))
svmRadial_rfe_pca_m <- train_rfe_pca_model(svm_model, training_data, validation_data, "svmRadial", "./ADMCI/svmRadial_model.rds", preProcess = c("center", "scale"))
svmPoly_rfe_pca_m <- train_rfe_pca_model(svm_model, training_data, validation_data, "svmPoly", "./ADMCI/svmPoly_model.rds", preProcess = c("center", "scale"))
svmLinearWeights_rfe_pca_m <- train_rfe_pca_model(svm_model, training_data, validation_data, "svmLinearWeights", "./ADMCI/svmLinearWeights_model.rds", preProcess = c("center", "scale"))
svmRadialWeights_rfe_pca_m <- train_rfe_pca_model(svm_model, training_data, validation_data, "svmRadialWeights", "./ADMCI/svmRadialWeights_model.rds", preProcess = c("center", "scale"))
rf_rfe_pca_m <- train_rfe_pca_model(rf_model, training_data, validation_data, "rf", "./ADMCI/rf_model.rds", preProcess = c("center", "scale"))
lr_rfe_pca_m <- train_rfe_pca_model(logistic_model, training_data, validation_data, "glm", "./ADMCI/lr_model.rds", prob = TRUE, family = "binomial", preProcess = c("center", "scale"))
knn_rfe_pca_m <- train_rfe_pca_model(knn_model, training_data, validation_data, "kknn", "./ADMCI/knn_model.rds", preProcess = c("center", "scale"))
gbm_rfe_pca_m <- train_rfe_pca_model(gbm_model, training_data, validation_data, "gbm", "./ADMCI/gbm_model.rds", verbose = FALSE, preProcess = c("center", "scale"))
xgb_rfe_pca_m <- train_rfe_pca_model(xgb_model, training_data, validation_data, "xgbTree", "./ADMCI/xgb_model.rds", preProcess = c("center", "scale"))
nb_rfe_pca_m <- train_rfe_pca_model(nb_model, training_data, validation_data, "naive_bayes", "./ADMCI/nb_model.rds", prob = TRUE, preProcess = c("center", "scale"))
dt_rfe_pca_m <- train_rfe_pca_model(dt_model, training_data, validation_data, "rpart", "./ADMCI/dt_model.rds", prob = TRUE, preProcess = c("center", "scale"))
nn_rfe_pca_m <- train_rfe_pca_model(nn_model, training_data, validation_data, "nnet", "./ADMCI/nn_model.rds", prob = TRUE, preProcess = c("center", "scale"))
# Rename column names
colnames(eval_metrics_rfe_pca) <- c("Model", "AUC", "MCC")
# Print the evaluation metrics for each model
print(eval_metrics_rfe_pca)
# Select the best model based on AUC
best_model <- eval_metrics_rfe_pca[which.max(eval_metrics_rfe_pca$AUC), "Model"]
print(paste("Best Model:", best_model))


#########################################
# Detect outliers 
#########################################
# Load necessary libraries
library(ggplot2)

# Subset the dataset with the desired columns
ADMCI_trainSet <- ADMCItrain_selected_numLabel[, c("Left.Angular.Gyrus", "Left.Hippocampus", "Left.Caudate", "Right.Angular.Gyrus", "Right.Lateral.Orbitofrontal.Gyrus", "Label")]

# Load necessary libraries
library(ggplot2)
library(tidyr)

# Specify the graphics device
options(device = "X11")

# Reshape the data into a long format
ADMCI_trainSet_long <- gather(ADMCI_trainSet, key = "Variable", value = "Value", -Label)

# Create a boxplot with high resolution and colors
boxplot <- ggplot(ADMCI_trainSet_long, aes(x = factor(Label), y = Value, fill = factor(Label))) +
  geom_boxplot(color = "black") +
  facet_wrap(~ Variable, scales = "free") +
  theme_bw() +
  labs(x = "Label", y = "Value") +
  ggtitle("Boxplot of Variables") +
  theme(plot.title = element_text(size = 14, face = "bold")) +
  theme(axis.title = element_text(size = 12)) +
  theme(axis.text = element_text(size = 10)) +
  theme(legend.position = "none") +
  theme(panel.grid = element_blank())

# Save the plot with high resolution and colors direADy to desk
ggsave("boxplot.png", plot = boxplot, width = 12, height = 8, dpi = 300)

# Reset the graphics device
dev.off()

###################################################################################

# Define function to test a model
test_model <- function(file_path, all_test_data, prob = FALSE) {
  # Load the model from a file
  model <- readRDS(file_path)
  test_data <- all_test_data
  
  # Predict using the model
  if (prob) {
    predictions_prob <- predict(model, newdata = test_data, type = "prob")
    predictions <- ifelse(predictions_prob[, "AD"] > 0.5, 1, 0)
  } else {
    predictions <- predict(model, newdata = test_data)
    predictions <- as.factor(ifelse(predictions == "AD", 1, 0))
  }
  # Calculate AUC and MCC
  print(predictions_prob)
  print(predictions)
  
  # Prepare for ROC and MCC calculations
  test_data$Label <- as.numeric(ifelse(test_data$Label == 'AD', 1, 0))
  predictions <- as.numeric(ifelse(predictions == 1, 1, 0))
  
  
  auc <- roc(test_data$Label, as.numeric(as.character(predictions)))$auc
  mcc <- mcc(preds = as.numeric(as.character(predictions)), actuals = test_data$Label)
  
  # Print evaluation metrics
  print(paste0("Test AUC=", auc, ", Test MCC=", mcc))
  
  
  # Return predictions
  return(predictions)
}

optimal_vars <- svm_model$rfe$optVariables
val_data <- validation_data[, c(optimal_vars, "Label")]

# Testing the svm model on the test data
svmLinear_test_preds <- test_model("./ADMCI/svm_model.rds", val_data, prob = TRUE )



##############
############
# Testing the model and creating the prediction csv file
# Test function to generate predictions
test_model <- function(model_file_path, all_test_data, rfe_model, ...) {
  # Load the model from a file
  model <- readRDS(model_file_path)
  print(ncol(all_test_data))
  print(rfe_model$rfe$optVariables)
  # Select the best features chosen by RFE
  optimal_vars <- rfe_model$rfe$optVariables
  test_data <- all_test_data[, optimal_vars]
  
  # Predict using the model
  predictions_prob <- predict(model, newdata = test_data, type = "prob")
  predictions <- ifelse(predictions_prob[, "AD"] > 0.5, 'AD', 'MCI') # updated
  
  # Create a dataframe for submission
  submission <- data.frame(ID = all_test_data$ID, 
                           Label = predictions, 
                           Probability = predictions_prob[, "AD"])
  
  # Write the dataframe to a csv file
  write.csv(submission, file = "0075710_NGIM_ADMCIres.csv", row.names = FALSE)
  
  # Create dataframe of selected features
  selected_features <- data.frame(Feature_Index = match(optimal_vars, names(all_test_data)))
  
  # Write the selected features dataframe to a csv file
  write.csv(selected_features, file = "0075710_NGIM_ADMCIfeat.csv", row.names = FALSE)
  
  return(submission)
}

#0075710_NGIM_ADMCIres.csv
#0075710_NGIM_ADMCIfeat.csv

test_data <- read.csv("ADMCItest.csv")
ncol(test_data)
submission <- test_model("./ADMCI/svm_model.rds", test_data, svm_model)


