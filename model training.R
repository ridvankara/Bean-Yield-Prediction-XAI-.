
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


# --- Stage 4.3: Statistical Significance (Bonferroni) ---

diffs <- diff(comparison)


bonferroni_summary <- summary(diffs, adjustment = "bonferroni")
print(bonferroni_summary)


# --- Stage 5: Taylor Diagram ---
library(plotrix)


preds_list <- list(
  SVM = predict(model_svm, test_scaled),
  ANN = predict(model_ann, test_scaled),
  RF = predict(model_rf, test_scaled),
  XGB = predict(model_xgb, test_scaled),
  GBM = predict(model_gbm, test_scaled)
)


taylor.diagram(test_scaled$x100_seed_weight, preds_list$SVM, col="red", pch=19, main="Taylor Diagram of Bean Seed Weight Prediction")
taylor.diagram(test_scaled$x100_seed_weight, preds_list$ANN, add=TRUE, col="blue", pch=19)
taylor.diagram(test_scaled$x100_seed_weight, preds_list$RF, add=TRUE, col="green", pch=19)
taylor.diagram(test_scaled$x100_seed_weight, preds_list$XGB, add=TRUE, col="orange", pch=19)

legend("topright", legend=names(preds_list[1:4]), fill=c("red", "blue", "green", "orange"))

# --- Stage 6: Bootstrap Analysis (Stability Check) ---
library(boot)


rsq_function <- function(formula, data, indices) {
  d <- data[indices,] # Örneklem seçimi
  fit <- lm(formula, data=d)
  return(summary(fit)$r.square)
}


boot_results <- boot(data = test_scaled, statistic = rsq_function, 
                     R = 1000, formula = x100_seed_weight ~ .)


plot(boot_results)
boot_ci <- boot.ci(boot_results, type="perc")
print(boot_ci) 


# --- Stage 7: AMMI / Biplot Analysis ---
if (!require("agricolae")) install.packages("agricolae"); library(agricolae)




ammi_model <- with(fas5_cleaned, AMMI(Genotype, x100_seed_weight, Bract_size, Seed_length)) 



plot(ammi_model, 0, 1, g_labels = "none") # PC1 vs Yield (Weight)
abline(h = 0, v = 0, lty = 2)


# --- Stage 8: XAI & Decision Support (SHAP & Waterfall) ---
if (!require("DALEX")) install.packages("DALEX"); library(DALEX)
if (!require("iBreakDown")) install.packages("iBreakDown"); library(iBreakDown)


explainer_svm <- explain(
  model = model_svm,
  data = train_scaled[, -which(names(train_scaled) == "x100_seed_weight")],
  y = train_scaled$x100_seed_weight,
  label = "SVM Model"
)
# --- Stage 9: Feature Importance (SVM) ---
library(caret)
library(ggplot2)


importance_svm <- varImp(model_svm, scale = FALSE)


importance_df <- as.data.frame(importance_svm$importance)
importance_df$Feature <- rownames(importance_df)

ggplot(importance_df, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "darkcyan") +
  coord_flip() +
  labs(title = "Feature Importance: Morphological Drivers of Seed Weight",
       x = "Morphological Traits", y = "Importance Score") +
  theme_minimal()


# --- Stage 5: Partial Dependence Plots (PDP) ---
if (!require("pdp")) install.packages("pdp"); library(pdp)


pdp_length <- partial(model_svm, pred.var = "Seed_length", plot = TRUE, rug = TRUE, 
                      plot.engine = "ggplot2") + ggtitle("PDP: Effect of Seed Length")

pdp_width <- partial(model_svm, pred.var = "Seed_width", plot = TRUE, rug = TRUE, 
                     plot.engine = "ggplot2") + ggtitle("PDP: Effect of Seed Width")

library(gridExtra)
grid.arrange(pdp_length, pdp_width, ncol = 2)

prediction_breakdown <- predict_parts(
  explainer = explainer_svm,
  new_observation = test_scaled[5, ],
  type = "break_down"
)
# --- Stage 10: Waterfall Analysis (XAI) ---
library(DALEX)
library(iBreakDown)


svm_exp <- explain(
  model = model_svm,
  data = train_scaled[, -which(names(train_scaled) == "x100_seed_weight")],
  y = train_scaled$x100_seed_weight,
  label = "SVM Decision Support"
)


prediction_breakdown <- predict_parts(
  explainer = svm_exp, 
  new_observation = test_scaled[5, ], 
  type = "break_down"
)


plot(prediction_breakdown) + 
  ggtitle("Waterfall Plot
)

plot(prediction_breakdown) + ggtitle("Waterfall Analysis: genotype ID 5")


