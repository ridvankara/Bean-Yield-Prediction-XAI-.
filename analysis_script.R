title: "beam4"
author: "rıdvan kara"
date: "2026-01-17"
output:
  pdf_document: default
  html_document: default

# --- Stage 1: Data Loading & Pre-processing ---

# Load necessary libraries
library(readr)
library(dplyr)
library(car)      # For VIF
library(caret)    # For Data Splitting and Normalization

# 1.1. Load Dataset (fas3.xlsx - Sayfa1.csv)
raw_data <- read_csv("fas3.xlsx - Sayfa1.csv")

# 1.2. Translate Column Names to English (Consistency for GitHub)
# The target variable (100 seed weight) is usually the last one
colnames(raw_data) <- c(
  "Terminal_leaflet_size", "Bract_size", "Flower_bud_length", 
  "Flower_bud_width", "Peduncle_length", "Internodes_with_first_flower", 
  "Flower_buds_in_cluster", "Pod_number_in_cluster", "Pod_length", 
  "Pod_width", "Pod_flesh_thickness", "Length_of_beak", 
  "Seed_number_in_pod", "Plant_height", "Number_of_internodes", 
  "Seed_emergence_time", "Inflorescence_time", "Seed_length", 
  "Seed_width", "Seed_height", "Hundred_seed_weight", "Harvest_time"
)

# 1.3. Remove Variable with High Missing Data (per Article Section 2.1)
# As per the article, 'first_pod_height' was removed. 
# If it's already not in your 22 variables, we proceed.
# Let's ensure the data is numeric
data_clean <- raw_data %>% mutate(across(everything(), as.numeric))

# 1.4. Check for Outliers using Cook's Distance (per Article Section 2.2.2)
# We use a simple linear model to calculate Cook's Distance
lm_model <- lm(Hundred_seed_weight ~ ., data = data_clean)
cooksd <- cooks.distance(lm_model)

# Threshold: 4/n (per Article)
n <- nrow(data_clean)
outliers <- which(cooksd > (4/n))
cat("Outliers detected at rows:", outliers)

# --- Stage 2: Multicollinearity (VIF) ---

# 2.1. Calculate VIF for all variables (Scenario 1)
vif_values <- vif(lm(Hundred_seed_weight ~ ., data = data_no_outliers))
print(vif_values)

# 2.2. Remove 'Number_of_internodes' due to high VIF (>10) [cite: 223, 247]
# This results in the 21-variable dataset (fas5 logic)
data_vif_optimized <- data_no_outliers %>% 
  select(-Number_of_internodes)

# Re-check VIF (Scenario 2)
vif_values_final <- vif(lm(Hundred_seed_weight ~ ., data = data_vif_optimized))
print(vif_values_final)


# --- Stage 1: Data Cleaning (Removing Specific Outliers) ---

# Packages check and load
if (!require("dplyr")) install.packages("dplyr"); library(dplyr)
if (!require("caret")) install.packages("caret"); library(caret)

# 1.1. Load your manually prepared 'fas5' dataset
# Assuming you loaded it as: fas5 <- read_csv("your_file.csv")
# Note: Ensure Hundred_seed_weight is the target variable name

# 1.2. Remove specific outlier genotypes provided by you
outliers_to_remove <- c(5, 15, 28, 69, 87, 93, 106, 110, 112, 116, 122, 125)
fas5_cleaned <- fas5[-outliers_to_remove, ]

cat("Initial rows: 125 | Final rows after outlier removal:", nrow(fas5_cleaned), "\n")


> set.seed(123)
> # --- Cross-Validation Setup ---
> train_ctrl <- trainControl(method = "cv", number = 10, savePredictions = "final")
> 
> # 1. SVM (Radial) - sigma: 0.001-0.1, C: 1-100
> svm_grid <- expand.grid(sigma = c(0.001, 0.01, 0.1), C = c(1, 10, 50, 100))
> model_svm <- train(x100_seed_weight ~ ., data = train_scaled, method = "svmRadial",
+                    trControl = train_ctrl, tuneGrid = svm_grid)
> 
> # 2. ANN (nnet) - size: 3-10, decay: 0.1-0.001
> ann_grid <- expand.grid(size = seq(3, 10, by = 2), decay = c(0.1, 0.01, 0.001))
> model_ann <- train(x100_seed_weight ~ ., data = train_scaled, method = "nnet",
+                    trControl = train_ctrl, tuneGrid = ann_grid, linout = TRUE, trace = FALSE)
> 
> # 3. Random Forest (RF) - mtry: 4, 8, 12, 16, ntree: 1000
> rf_grid <- expand.grid(mtry = c(4, 8, 12, 16))
> model_rf <- train(x100_seed_weight ~ ., data = train_scaled, method = "rf",
+                   trControl = train_ctrl, tuneGrid = rf_grid, ntree = 1000)
> 
> # 4. XGBoost - nrounds: 500-1000, depth: 4-8, eta: 0.01-0.05
> xgb_grid <- expand.grid(nrounds = c(500, 1000), max_depth = c(4, 6, 8),
+                         eta = c(0.01, 0.05), gamma = 0, colsample_bytree = 0.8,
+                         min_child_weight = 1, subsample = 0.8)
> model_xgb <- train(x100_seed_weight ~ ., data = train_scaled, method = "xgbTree",
+                    trControl = train_ctrl, tuneGrid = xgb_grid)

> 
> # 5. GBM - depth: 3-7, shrinkage: 0.01-0.05
> gbm_grid <- expand.grid(interaction.depth = c(3, 5, 7), n.trees = c(100, 500),
+                         shrinkage = c(0.01, 0.05), n.minobsinnode = 10)
> model_gbm <- train(x100_seed_weight ~ ., data = train_scaled, method = "gbm",
+                    trControl = train_ctrl, tuneGrid = gbm_grid, verbose = FALSE)
> # 6. LightGBM - num_leaves: 15-31, learning_rate: 0.05
> # Matris formatına hazırlık
> dtrain_lgb <- lgb.Dataset(data = as.matrix(train_scaled[,-which(names(train_scaled)=="x100_seed_weight")]), 
+                           label = train_scaled$x100_seed_weight)
> 
> lgb_params <- list(objective = "regression", metric = "rmse", 
+                    learning_rate = 0.05, num_leaves = 31, min_data_in_leaf = 10)
> 
> model_lgbm <- lgb.train(params = lgb_params, data = dtrain_lgb, nrounds = 100, verbose = -1)
> # Metrikleri hesaplayan fonksiyon
> get_metrics <- function(model, test_data, target_col) {
+   preds <- predict(model, test_data)
+   actual <- test_data[[target_col]]
+   
+   r2 <- cor(actual, preds)^2
+   rmse <- sqrt(mean((actual - preds)^2))
+   mae <- mean(abs(actual - preds))
+   
+   return(c(R2 = r2, RMSE = rmse, MAE = mae))
+ }
> 
> # Modellerin Test Seti Performansı
> results_table <- data.frame(
+   SVM = get_metrics(model_svm, test_scaled, "x100_seed_weight"),
+   ANN = get_metrics(model_ann, test_scaled, "x100_seed_weight"),
+   RF  = get_metrics(model_rf, test_scaled, "x100_seed_weight"),
+   XGB = get_metrics(model_xgb, test_scaled, "x100_seed_weight"),
+   GBM = get_metrics(model_gbm, test_scaled, "x100_seed_weight")
+ )
> 
> print(t(results_table))



