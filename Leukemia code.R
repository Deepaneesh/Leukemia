# Library ###########################
library(dplyr)
library(tidyverse)
library(rsample)
library(caret)
library(Metrics) # for confusion matrix
library(catboost)
library(randomForest)
library(caret)
library(dplyr)
library(ggplot2)
library(xgboost)
library(tidymodels)
library(dplyr)
library(caret)

# data Cleaning #######################
leukemia=read.csv(file.choose(),header=T)
colnames(leukemia)
data=leukemia
data=na.omit(data)
data <- data[1:402, ]
data %>% View()
str(data)
data=data %>% mutate(Gender=as.factor(Gender),
                     Genetic_Mutation=as.factor(Genetic_Mutation),
                     Family_History=as.factor(Family_History),
                     Smoking_Status=as.factor(Smoking_Status),
                     Alcohol_Consumption=as.factor(Alcohol_Consumption),
                     Radiation_Exposure=as.factor(Radiation_Exposure),
                     Infection_History=as.factor(Infection_History),
                     Chronic_Illness=as.factor(Chronic_Illness),
                     Immune_Disorders=as.factor(Immune_Disorders),
                     Socioeconomic_Status=as.factor(Socioeconomic_Status),
                     Urban_Rural=as.factor(Urban_Rural),
                     Leukemia_Status=as.factor(Leukemia_Status)
                     )
str(data)
data$Gender
table(data$Leukemia_Status)
summary(data)
# Data Visualization ##################

data %>% group_by(Gender) %>% summarise(Count=n()) %>% mutate(Percentage=Count/sum(Count)*100) %>% 
  ggplot(aes(x="",y=Percentage,fill=Gender))+
  geom_bar(stat="identity",width=1)+
  coord_polar("y",start=0)+
  labs(title = "  GENDER")+
  geom_text(aes(label=paste0(Gender,"-",round(Percentage),"%")),position=position_stack(vjust=0.5) ,fontface = "bold")+theme_void()+ 
  theme(plot.title = element_text(face = "bold",
    hjust = 0.5,size = 17),legend.position = "none")


data %>% group_by(Leukemia_Status) %>% summarise(Count=n()) %>% mutate(Percentage=Count/sum(Count)*100) %>%
  ggplot(aes(x="",y=Percentage,fill=Leukemia_Status))+
  geom_bar(stat="identity",width=1)+coord_polar("y",start=0)+
  labs(title = " LEUKEMIA STATUS",
       fill="Leukemia Status")+
  geom_text(aes(label=paste0(Leukemia_Status,"-",round(Percentage),"%")),position=position_stack(vjust=0.5),fontface = "bold")+theme_void()+
  theme(plot.title = element_text(face = "bold",
    hjust = 0.5,,size = 17),legend.position = "none")

data %>% group_by(Genetic_Mutation) %>% summarise(Count=n()) %>% 
  ggplot(aes(x=Genetic_Mutation,y=Count,fill=Genetic_Mutation))+
  geom_bar(stat="identity",width=0.5)+
  labs(title = "GENETIC MUTATION",
       x="Genetic Mutation",
       y="Percentage",
       fill="Genetic Mutation")+theme_minimal()+
  theme(plot.title = element_text(face = "bold",
    hjust = 0.5,size = 17),legend.position = "none")+geom_text(aes(label=Count),vjust=-0.5,fontface = "bold")

data %>% group_by(Socioeconomic_Status) %>% summarise(Count=n()) %>% 
  ggplot(aes(x=Socioeconomic_Status,y=Count,fill=Socioeconomic_Status))+
  geom_bar(stat="identity",width=0.5)+
  labs(title = "SOCIOECONOMIC STATUS",
       x="Socioeconomic Status",
       y="Percentage",
       fill="Socioeconomic Status")+theme_minimal()+
  theme(plot.title = element_text(face = "bold",
    hjust = 0.5,size = 17))+geom_text(aes(label=Count),vjust=-0.5,fontface = "bold")

data %>% group_by(Socioeconomic_Status) %>% summarise(Count=n()) %>% mutate(Percentage=Count/sum(Count)*100) %>% 
  ggplot(aes(x="",y=Percentage,fill=Socioeconomic_Status))+
  geom_bar(stat="identity",width=1)+
  coord_polar("y",start=0)+
  labs(title = "SOCIOECONOMIC STATUS")+
  geom_text(aes(label=paste0(Socioeconomic_Status,"-",round(Percentage),"%")),position=position_stack(vjust=0.5),fontface = "bold")+theme_void()+ 
  theme(plot.title = element_text(face = "bold",
    hjust = 0.5,,size = 17),legend.position = "none")
# data Splitting ######################
set.seed(123)
data_split=initial_split(data,prop=0.7,strata=Leukemia_Status)
train_data=training(data_split)
test_data=testing(data_split)

# Model Building 
## 1.Logistic Regression===================

model1=glm(Leukemia_Status~.,data=train_data,family=binomial)
summary(model1)
predtarin1=predict(model1,train_data,type="response")
predtest1=predict(model1,test_data,type="response")

predtrain1=ifelse(predtarin1>0.5,"Positive","Negative") %>% as.factor()
predtest1=ifelse(predtest1>0.5,"Positive","Negative") %>% as.factor()

confusionMatrix(predtrain1,train_data$Leukemia_Status)
levels(train_data$Leukemia_Status)

## 2.CatBoost=============================

# Define categorical features
cat_features <- c("Genetic_Mutation", "Family_History", "Smoking_Status", "Alcohol_Consumption",
                  "Radiation_Exposure", "Infection_History", "Chronic_Illness", 
                  "Immune_Disorders", "Socioeconomic_Status", "Urban_Rural")

levels(train_data$Leukemia_Status)
# Convert target variable to numeric (0,1) for CatBoost
data1 <- data %>% mutate(Leukemia_Status = as.integer(as.factor(Leukemia_Status)) - 1)

# Split Data (Stratified by Leukemia_Status)
set.seed(123)
data_split1 <- initial_split(data1, prop = 0.7, strata = Leukemia_Status)
train_data1 <- training(data_split1)
test_data1 <- testing(data_split1)

# Convert to CatBoost Pool Format
train_pool <- catboost.load_pool(
  data = train_data1 %>% select(-Leukemia_Status),
  label = train_data1$Leukemia_Status,
  cat_features = which(names(train_data1) %in% cat_features)
)

test_pool <- catboost.load_pool(
  data = test_data1 %>% select(-Leukemia_Status),
  label = test_data1$Leukemia_Status,
  cat_features = which(names(test_data1) %in% cat_features)  # FIXED indexing issue
)

# Default CatBoost parameters
params <- list(
  loss_function = "Logloss",  # Binary classification
  iterations = 1000,          # Number of boosting rounds
  depth = 6,                  # Default tree depth
  learning_rate = 0.03,        # Default learning rate
  verbose = 100                # Print progress every 100 iterations
)

# Train the CatBoost Model
model <- catboost.train(train_pool, params = params)

# Predict Class (0 or 1)
preds <- catboost.predict(model, test_pool, prediction_type = "Class")

# Convert to factor for evaluation
preds <- factor(preds)
true_labels <- factor(test_data$Leukemia_Status)

# Releveling 
levels(preds)=levels(train_data$Leukemia_Status)
# Confusion Matrix & Accuracy
confusionMatrix(preds, true_labels) 

## 3.XG Boost=============================
library(xgboost)
library(tidymodels)
library(dplyr)
library(caret)
levels(train_data$Leukemia_Status)
# Convert categorical variables to numeric
data2 <- data %>% mutate(across(everything(), as.numeric))
data2$Leukemia_Status <- as.numeric(data2$Leukemia_Status) - 1  # Convert target to 0 & 1

# Split data
set.seed(123)
data_split2 <- initial_split(data2, prop = 0.7, strata = Leukemia_Status)
train_data2 <- training(data_split2)
test_data2 <- testing(data_split2)

# Convert to DMatrix (XGBoost format)
dtrain <- xgb.DMatrix(data = as.matrix(train_data2 %>% select(-Leukemia_Status)), 
                      label = train_data2$Leukemia_Status)
dtest <- xgb.DMatrix(data = as.matrix(test_data2 %>% select(-Leukemia_Status)), 
                     label = test_data2$Leukemia_Status)

# Define a grid of hyperparameters
grid <- expand.grid(
  max_depth = c(3, 6, 9),    
  eta = c(0.01, 0.1, 0.3),   
  nrounds = c(50, 100, 150)  
)

# Use tidymodels' tuning without a loop
results <- grid %>%
  rowwise() %>%
  mutate(
    model = list(
      xgb.train(
        params = list(
          objective = "binary:logistic",  # Binary classification
          max_depth = max_depth,    
          eta = eta,
          eval_metric = "logloss"  # Evaluation metric
        ),
        data = dtrain,
        nrounds = nrounds,
        verbose = 0
      )
    ),
    preds = list(predict(model[[1]], dtest)),  # Make predictions
    predicted_classes = list(ifelse(preds[[1]] > 0.5, 1, 0)),  # Convert probs to classes
    accuracy = mean(predicted_classes[[1]] == test_data2$Leukemia_Status)  # Compute accuracy
  ) %>%
  select(max_depth, eta, nrounds, accuracy)  # Keep only relevant columns

# Print results
print(results)
results %>% arrange(desc(accuracy))
best_result <- results[which.max(results$accuracy), ]
print(paste("Best Accuracy:", round(best_result$accuracy, 4)))
print(best_result)


# Predictions
best_model <- xgb.train(
  params = list(
    objective = "binary:logistic",
    max_depth = best_result$max_depth,
    eta = best_result$eta,
    eval_metric = "logloss"
  ),
  data = dtrain,
  nrounds = best_result$nrounds,
  verbose = 0
)
test_predictions <- predict(best_model, dtest)
test_pred_classes <- ifelse(test_predictions > 0.5,"Positive", "Negative") %>% as.factor()
k= test_data2$Leukemia_Status %>% as.factor()
levels(k)=levels(data$Leukemia_Status)
# Confusion Matrix & Accuracy
confusionMatrix(test_pred_classes,k )

## 4.Random Forest=========================
library(randomForest)
library(caret)
library(dplyr)
library(ggplot2)

# Check levels of target variable
levels(train_data3$Leukemia_Status)

# Splitting data
set.seed(123)
data_split3 <- initial_split(data, prop = 0.7, strata = Leukemia_Status)
train_data3 <- training(data_split3)
test_data3 <- testing(data_split3)

# Define training control with 5-fold cross-validation
train_control1 <- trainControl(method = "cv", number = 5)

# Define tuning grid for mtry (number of predictors sampled at each split)
tune_grid1 <- expand.grid(mtry = c(1:5, seq(6, 18, by = 2)))

# Train the Random Forest model
rf_model1 <- train(
  Leukemia_Status ~ ., 
  data = train_data3, 
  method = "rf", 
  trControl = train_control1, 
  tuneGrid = tune_grid1
)

# Print model results
print(rf_model1)

# Convert results to dataframe
return1 <- rf_model1$results %>% as.data.frame()

# Select and filter based on mtry values
return1 %>%
  select(mtry, Accuracy, Kappa)  %>%
  arrange(desc(Accuracy))

return1 %>%
  select(mtry, Accuracy, Kappa) %>%
  filter(mtry == 10 | mtry == 8 | mtry == 12) %>%
  arrange(desc(Accuracy))

# Plot model performance across different mtry values
ggplot(return1, aes(x = mtry, y = Accuracy)) +
  geom_line(color = "Blue") +
  geom_point() +
  labs(
    title = "Random Forest Model Performance",
    x = "mtry",
    y = "Accuracy"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_x_continuous(breaks = seq(1, 18, by = 2)) +
  scale_y_continuous(limits = c(0.65, 0.9), breaks = seq(0.65, 0.9, by = 0.05))

# Train Random Forest model with best mtry (e.g., 16)
rf_model11 <- randomForest(Leukemia_Status ~ ., data = train_data3, mtry = 16, importance = TRUE)
print(rf_model11)

# Predict on test data
test_pred2 <- predict(rf_model11, newdata = test_data3)

# Predict on training data
train_pred2 <- predict(rf_model11, newdata = train_data3)

# Confusion matrix for test data
conf_matrix_test <- confusionMatrix(test_pred2, test_data3$Leukemia_Status)
print(conf_matrix_test)

# Confusion matrix for train data
conf_matrix_train <- confusionMatrix(train_pred2, train_data3$Leukemia_Status)
print(conf_matrix_train)

## 5.Glmnet===============================
library(glmnet)

# Split data
set.seed(123)
data3 <- data %>% mutate(across(everything(), as.numeric))
data3$Leukemia_Status=data$Leukemia_Status
data3
data_split4 <- initial_split(data3, prop = 0.7, strata = Leukemia_Status)
train_data4 <- training(data_split4)
test_data4 <- testing(data_split4)

# Load required libraries
library(glmnet)
library(tidymodels)

# Load required libraries
library(glmnet)
library(tidymodels)
library(caret)

# Convert data to matrix format required by glmnet
x_train <- model.matrix(Leukemia_Status ~ ., data = train_data4)[, -1]  # Remove intercept column
y_train <- train_data4$Leukemia_Status  # Response variable (binary: 0/1)

x_test <- model.matrix(Leukemia_Status ~ ., data = test_data4)[, -1]
y_test <- test_data4$Leukemia_Status  

# Define alpha grid (elastic net mixing parameter)
alpha_grid <- seq(0, 1, by = 0.1)  # Grid search from 0 (ridge) to 1 (lasso)

# Create an empty data.frame to store results
cv_results_df <- data.frame(
  alpha = numeric(),
  best_lambda = numeric(),
  min_deviance = numeric(),
  Testaccuracy = numeric(),
  Trainaccuracy = numeric()
)

# Perform cross-validation for each alpha and store results in data.frame
set.seed(123)
for (alpha_value in alpha_grid) {
  cv_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = alpha_value, nfolds = 10)
  best_lambda <- cv_model$lambda.min
  
  # Train best model
  best_model <- glmnet(x_train, y_train, family = "binomial", alpha = alpha_value, lambda = best_lambda)
  
  # Predict on test and train data
  test_predict <- predict(best_model, newx = x_test, type = "class") %>% factor()
  train_predict <- predict(best_model, newx = x_train, type = "class") %>% factor()
  
  # Compute confusion matrices
  test_accuracy <- confusionMatrix(test_predict, y_test)$overall["Accuracy"]
  train_accuracy <- confusionMatrix(train_predict, y_train)$overall["Accuracy"]
  
  # Append results to the data frame
  cv_results_df <- rbind(cv_results_df, data.frame(
    alpha = alpha_value,
    best_lambda = best_lambda,  # Optimal lambda
    min_deviance = min(cv_model$cvm),  # Cross-validation error
    Testaccuracy = test_accuracy,
    Trainaccuracy = train_accuracy
  ))
}

cv_results_df=cv_results_df %>% arrange(desc(Trainaccuracy),desc(Testaccuracy))
cv_results_df

# Fitting the best model 
best_alpha= cv_results_df$alpha[1]

# glmnet model with best alpha
model=cv.glmnet(x_train, y_train, family = "binomial", alpha = best_alpha)
best_lambda <- model$lambda.min
plot(model)
best_lambda

best_model <- glmnet(x_train, y_train, family = "binomial", alpha = best_alpha, lambda = best_lambda)
options(scipen = 999) 
coef(best_model)
coef(best_model) %>% plot()
# Predict on test data
pred5=predict(best_model, newx = x_test, type = "class") %>% factor()

# confusion matrix
confusionMatrix(pred5, y_test)



# Load necessary libraries
library(ggplot2)
library(glmnet)
library(tibble)
library(dplyr)

# Extract coefficients and convert to a data frame
coef_df <- as.data.frame(as.matrix(coef(best_model)))
coef_df <- coef_df %>%
  rownames_to_column(var = "Feature") %>%  # Convert row names (features) into a column
  rename(Coefficient = s0)  # 's0' refers to the single lambda used

# Remove intercept (optional)
coef_df <- coef_df %>% filter(Feature != "(Intercept)")

# Plot using ggplot
ggplot(coef_df, aes(x = reorder(Feature, Coefficient), y = Coefficient, fill = Coefficient > 0)) +
  geom_bar(stat = "identity", show.legend = FALSE) +  # Bar plot
  coord_flip() +  # Flip for better readability
  labs(title = "Feature Coefficients from glmnet Model",
       x = "Features",
       y = "Coefficient Value") +
  scale_fill_manual(values = c("red", "blue")) +  # Color negative (red) & positive (blue)
  theme_minimal() + theme(panel.grid.minor = element_line(linetype = "dashed"))

