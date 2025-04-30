df <- fread("UrbanSound8K_7days_HourNormalized.csv", stringsAsFactors = T)
str(df)
levels(df$class)
levels(df$district)


############## Predicting areas at risk for noise complaints #####################
library(data.table)
library(dplyr)
library(caret)
library(pROC)
library(ggplot2)
library(rpart)          # Decision Trees
library(randomForest)   # Random Forest
library(xgboost)        # XGBoost
library(e1071)          # SVM, Naive Bayes
library(Metrics)        # For MSE, RMSE, MAE if needed
library(MLmetrics)      # For F1 Score


############################### Preprocessing ###################################
# Simulate the real-world variation
df_noise_complaints <- df
df_noise_complaints$lasting <- df_noise_complaints$end - df_noise_complaints$start

df_noise_complaints$decibel_noisy <- df_noise_complaints$decibel + rnorm(nrow(df_noise_complaints), mean=0, sd=1.5)
df_noise_complaints$decibel_noisy <- pmin(pmax(df_noise_complaints$decibel_noisy, 60), 130)

head(df_noise_complaints[, c('class', 'decibel', 'decibel_noisy')])

# Plot the distribution
ggplot(df_noise_complaints, aes(x = decibel_noisy)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "skyblue", color = "black") +
  geom_density(color = "darkblue", size = 1) +
  labs(title = "Distribution of Decibel Noisy",
       x = "Decibel (with Noise)",
       y = "Density") +
  theme_minimal()

# Q-Q Plot
qqnorm(df_noise_complaints$decibel_noisy, main = "Q-Q Plot of Decibel Noisy")
qqline(df_noise_complaints$decibel_noisy, col = "red", lwd = 2)

# Generate a realistic and reasonable complaint report
# Start with a base complaint probability (low)
df_noise_complaints$complaint_prob <- 0.05
# 1. Overnight noises (after 22 or before 5) → add 0.20
df_noise_complaints$complaint_prob <- df_noise_complaints$complaint_prob +
  ifelse(df_noise_complaints$hour >= 22 | df_noise_complaints$hour <= 5, 0.20, 0)
# 2. Noise lasting more than 2 seconds → add 0.15
df_noise_complaints$complaint_prob <- df_noise_complaints$complaint_prob +
  ifelse(df_noise_complaints$lasting > 2, 0.15, 0)
# 3. Weekday (Mon.-Fri.) vs Weekend → add 0.10
df_noise_complaints$complaint_prob <- df_noise_complaints$complaint_prob +
  ifelse(df_noise_complaints$day %in% 1:5, 0.10, 0)
# 4. Loud noise class types → add 0.15
df_noise_complaints$complaint_prob <- df_noise_complaints$complaint_prob +
  ifelse(df_noise_complaints$class %in% c("car_horn", "drilling", "engine_idling", "gun_shot", "jackhammer", "street_music"), 0.15, 0)
# 5. Decibel noisy thresholds
df_noise_complaints$complaint_prob <- df_noise_complaints$complaint_prob +
  ifelse(df_noise_complaints$decibel_noisy > 70, 0.20, 
         ifelse(df_noise_complaints$decibel_noisy > 50, 0.10, 0))

df_noise_complaints$complaint_prob <- pmin(df_noise_complaints$complaint_prob, 0.95)

# Simulate whether a complaint actually occurs
# Randomly draw a number between 0-1; if below complaint_prob → complaint happens
df_noise_complaints$high_noise <- rbinom(nrow(df_noise_complaints), size = 1, prob = df_noise_complaints$complaint_prob)

head(df_noise_complaints[, c("decibel", "decibel_noisy", "lasting", "hour", "day", "class", "complaint_prob", "high_noise")])


############################### Train/Test Split #################################
set.seed(123)
train_index <- createDataPartition(df_noise_complaints$high_noise, p = 0.75, list = FALSE)
train_data <- df_noise_complaints[train_index, ]
test_data <- df_noise_complaints[-train_index, ]

# Convert high_noise to factors for classification models
train_data$high_noise <- factor(ifelse(train_data$high_noise == 1, "yes", "no"))
test_data$high_noise <- factor(ifelse(test_data$high_noise == 1, "yes", "no"))

### Cross-Validation Control
control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

### Features
features <- c("decibel_noisy", "lasting", "hour", "day", "class", "latitude", "longitude")
formula <- as.formula(paste("high_noise ~", paste(features, collapse = " + ")))


############################### Model Training ###################################
### Logistic Regression
set.seed(123)
model_logit <- train(formula, data = train_data, method = "glm", family = "binomial", trControl = control, metric = "ROC")

### Decision Tree
set.seed(123)
model_tree <- train(formula, data = train_data, method = "rpart", trControl = control, metric = "ROC")

### Random Forest
set.seed(123)
model_rf <- train(formula, data = train_data, method = "rf", trControl = control, metric = "ROC", tuneLength = 5)

### XGBoost
set.seed(123)
model_xgb <- train(formula, data = train_data, method = "xgbTree", trControl = control, metric = "ROC", tuneLength = 5)

### SVM
set.seed(123)
model_svm <- train(formula, data = train_data, method = "svmRadial", trControl = control, metric = "ROC", tuneLength = 5)

### Naive Bayes
set.seed(123)
model_nb <- train(formula, data = train_data, method = "naive_bayes", trControl = control, metric = "ROC", tuneLength = 5)


########################## Prediction and Evaluation ############################
# Build a function to run the result
evaluate_model <- function(model, test_data, model_name) {
  probs <- predict(model, newdata = test_data, type = "prob")[, "yes"]
  preds <- predict(model, newdata = test_data)
  
  cm <- confusionMatrix(preds, test_data$high_noise)
  roc_curve <- roc(response = test_data$high_noise, predictor = probs, levels = rev(levels(test_data$high_noise)))
  
  accuracy <- cm$overall['Accuracy']
  auc_score <- roc_curve$auc  # Correct: directly pull AUC from roc_curve object
  f1 <- F1_Score(y_pred = preds, y_true = test_data$high_noise, positive = "yes")
  
  list(Model = model_name, Accuracy = accuracy, AUC = auc_score, F1 = f1, ConfMatrix = cm)
}

# Collect results
results <- list()
results[[1]] <- evaluate_model(model_logit, test_data, "Logistic Regression")
results[[2]] <- evaluate_model(model_tree, test_data, "Decision Tree")
results[[3]] <- evaluate_model(model_rf, test_data, "Random Forest")
results[[4]] <- evaluate_model(model_xgb, test_data, "XGBoost")
results[[5]] <- evaluate_model(model_svm, test_data, "SVM")
results[[6]] <- evaluate_model(model_nb, test_data, "Naive Bayes")

summary_results <- do.call(rbind, lapply(results, function(x) {
  data.frame(Model = x$Model, Accuracy = round(as.numeric(x$Accuracy), 4),
             AUC = round(as.numeric(x$AUC), 4),
             F1_Score = round(as.numeric(x$F1), 4))
}))

print(summary_results)


plot(roc(response = test_data$high_noise, predictor = predict(model_logit, newdata = test_data, type = "prob")[, "yes"]), col = "blue", main = "ROC Curves", lwd = 2)
lines(roc(response = test_data$high_noise, predictor = predict(model_tree, newdata = test_data, type = "prob")[, "yes"]), col = "green", lwd = 2)
lines(roc(response = test_data$high_noise, predictor = predict(model_rf, newdata = test_data, type = "prob")[, "yes"]), col = "red", lwd = 2)
lines(roc(response = test_data$high_noise, predictor = predict(model_xgb, newdata = test_data, type = "prob")[, "yes"]), col = "purple", lwd = 2)
lines(roc(response = test_data$high_noise, predictor = predict(model_svm, newdata = test_data, type = "prob")[, "yes"]), col = "orange", lwd = 2)
lines(roc(response = test_data$high_noise, predictor = predict(model_nb, newdata = test_data, type = "prob")[, "yes"]), col = "black", lwd = 2)
legend("bottomright", legend = c("Logistic", "Tree", "RF", "XGB", "SVM", "NB"),
       col = c("blue", "green", "red", "purple", "orange", "black"), lwd = 2)

