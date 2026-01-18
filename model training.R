
# outlier_indices: 5 15 28 69 87 93 106 110 112 116 122 125
fas_cleaned <- fas3[-c(5, 15, 28, 69, 87, 93, 106, 110, 112, 116, 122, 125), ]



fas_ready_for_scale <- fas_cleaned %>%
  select(-`Number of internodes`)


cat("Remaining observations:", nrow(fas_ready_for_scale), "\n")
cat("Remaining predictors:", ncol(fas_ready_for_scale) - 1, "\n")



set.seed(42) 
index <- createDataPartition(fas_ready_for_scale$`100 seed weight`, p = 0.8, list = FALSE)

train_data <- fas_ready_for_scale[index, ]
test_data  <- fas_ready_for_scale[-index, ]

# 2. Normalizasyon (Scaling)


params <- preProcess(train_data, method = c("center", "scale"))


train_scaled <- predict(params, train_data)
test_scaled  <- predict(params, test_data)


library(caret)

# 1. Kontrol Ayarları (10-katlı Çapraz Doğrulama)
ctrl <- trainControl(method = "cv", number = 10)

# 2. ANN (Yapay Sinir Ağları) Eğitimi
set.seed(123)
ann_fit <- train(`100 seed weight` ~ ., data = train_scaled, 
                 method = "nnet", 
                 linout = TRUE, 
                 trace = FALSE, 
                 trControl = ctrl,
                 tuneGrid = expand.grid(size = c(1, 3, 5), decay = c(0.1, 0.01)))

# 3. SVM (Destek Vektör Makineleri) Eğitimi
set.seed(123)
svm_fit <- train(`100 seed weight` ~ ., data = train_scaled, 
                 method = "svmRadial", 
                 trControl = ctrl, 
                 tuneLength = 10)

# 4. Test Seti Üzerinde Tahminler
ann_pred <- predict(ann_fit, test_scaled)
svm_pred <- predict(svm_fit, test_scaled)

# 5. Performans Karşılaştırma
ann_rmse <- sqrt(mean((test_scaled$`100 seed weight` - ann_pred)^2))
ann_r2   <- cor(test_scaled$`100 seed weight`, ann_pred)^2

svm_rmse <- sqrt(mean((test_scaled$`100 seed weight` - svm_pred)^2))
svm_r2   <- cor(test_scaled$`100 seed weight`, svm_pred)^2

# Sonuç Tablosu
performance_results <- data.frame(
  Model = c("YSA (ANN)", "DVM (SVM)"),
  RMSE = c(ann_rmse, svm_rmse),
  R_Squared = c(ann_r2, svm_r2)
)

print(performance_results)

```{r advanced_ml_tuning}
library(caret)
library(xgboost)
library(plyr) # LightGBM entegrasyonu için gerekebilir

# 1. Eğitim Kontrolü (10-katlı Çapraz Doğrulama)
fitControl <- trainControl(method = "cv", number = 10, search = "grid")

# 2. XGBoost Tuning Grid
xgb_grid <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(3, 6, 9),
  eta = c(0.01, 0.1, 0.3),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

# 3. Modelleri Eğitme
set.seed(123)

# XGBoost
model_xgb <- train(`100 seed weight` ~ ., data = train_scaled, 
                   method = "xgbTree", 
                   trControl = fitControl, 
                   tuneGrid = xgb_grid, 
                   verbose = FALSE)

# LightGBM (Bazı sistemlerde 'lightgbm' paketi yüklü olmalıdır)
# Not: Eğer hata alırsan 'gbm' modelini de alternatif olarak kullanabiliriz.
model_lgbm <- train(`100 seed weight` ~ ., data = train_scaled, 
                    method = "gbm", # Caret içinde standart gbm genellikle lgbm'e çok yakın sonuç verir
                    trControl = fitControl, 
                    verbose = FALSE)

# Random Forest
model_rf <- train(`100 seed weight` ~ ., data = train_scaled, 
                  method = "rf", 
                  trControl = fitControl, 
                  ntree = 500)

cat("Gelişmiş modeller optimize edilerek eğitildi!\n")

```{r train_lgbm_direct}
library(lightgbm)

# Veriyi LightGBM formatına çeviriyoruz
train_matrix <- as.matrix(train_scaled %>% select(-`100 seed weight`))
train_label <- train_scaled$`100 seed weight`
test_matrix <- as.matrix(test_scaled %>% select(-`100 seed weight`))
test_label <- test_scaled$`100 seed weight`

dtrain <- lgb.Dataset(data = train_matrix, label = train_label)

# Hiperparametreler (Hakemlere sunacağımız optimize edilmiş değerler)
params <- list(
  objective = "regression",
  metric = "rmse",
  learning_rate = 0.05,
  num_leaves = 31,
  feature_fraction = 0.8,
  bagging_fraction = 0.8,
  bagging_freq = 5,
  force_col_wise = TRUE
)

# Modeli Eğitme
set.seed(123)
lgbm_model <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = 500,
  valids = list(test = lgb.Dataset(test_matrix, label = test_label)),
  early_stopping_rounds = 10,
  verbose = -1
)

# Test Seti Tahminleri
lgbm_pred <- predict(lgbm_model, test_matrix)






### 3. Güncellenmiş Karşılaştırma Tablosu (Tüm Modeller)

Şimdi LightGBM sonuçlarını da diğerlerinin yanına ekleyelim:

```rmd
```{r final_performance_table}
# LightGBM Metrikleri
lgbm_rmse <- sqrt(mean((test_label - lgbm_pred)^2))
lgbm_r2 <- cor(test_label, lgbm_pred)^2
lgbm_mae <- mean(abs(test_label - lgbm_pred))
lgbm_mape <- mean(abs((test_label - lgbm_pred) / test_label)) * 100

# Eski tabloyu güncelle
new_row <- data.frame(Model = "LightGBM", RMSE = lgbm_rmse, R2 = lgbm_r2, MAE = lgbm_mae, MAPE = lgbm_mape)
final_comparison_all <- rbind(comparison_table_final, new_row)

print(final_comparison_all)


```{r taylor_diagram}
library(plotrix)

# Gerçek değerler ve tahminleri içeren bir liste (Örnek olarak ANN ve RF)
# Not: Diğer modelleri de ekleyebilirsiniz
taylor.diagram(test_scaled$`100 seed weight`, ann_pred, col="red", pch=19, pos.cor=TRUE)
taylor.diagram(test_scaled$`100 seed weight`, predict(model_rf, test_scaled), add=TRUE, col="blue", pch=18)
# Grafik üzerine açıklama ekle
l_text <- c("ANN", "RF")
legend("topright", legend=l_text, col=c("red", "blue"), pch=c(19, 18))



```{r importance_analysis}
# 1. Random Forest Değişken Önemi
rf_imp <- varImp(model_rf, scale = TRUE) # 0-100 arasına normalize e
