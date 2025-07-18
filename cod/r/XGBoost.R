library(tidyverse)
library(xgboost)
library(caret)
library(readxl)
library(Metrics)
library(moments)
library(tseries)
library(hydroGOF)
library(hydroTSM)

# Función para cargar los datos

xg_cargar_datos <- function(ruta, var_obj){
  base <- read.csv(ruta)
  
  if (var_obj %in% names(base)) {
    base <- base[!is.na(base[[var_obj]]), ]
  } 
  return(base)
}

# Función para realizar la partición de los datos

xg_particion_datos <- function(datos, var_obj, tam_split = 0.65){
  n <- nrow(datos)
  corte <- floor(tam_split * n)
  
  train_data <- datos[1:corte, ]
  test_data <- datos[(corte+1):n, ]
  
  train_y <- train_data[[var_obj]]
  test_y <- test_data[[var_obj]]
  
  train_x <- train_data[, !(names(train_data)) %in% var_obj]
  test_x <- test_data[, !(names(test_data)) %in% var_obj]
  
  dummies <- dummyVars("~.", data = train_x)
  train_x <- predict(dummies, newdata = train_x)
  test_x <- predict(dummies, newdata = test_x)
  
  missing_cols <- setdiff(colnames(train_x), colnames(test_x))
  if(length(missing_cols) > 0) {
    test_x <- cbind(test_x, matrix(0, nrow = nrow(test_x), ncol = length(missing_cols)))
    colnames(test_x)[(ncol(test_x)-length(missing_cols)+1):ncol(test_x)] <- missing_cols
    test_x <- test_x[, colnames(train_x)]
  }
  
  train_x <- as.matrix(train_x)
  test_x <- as.matrix(test_x)
  dtrain <- xgb.DMatrix(data = train_x, label = train_y)
  dtest <- xgb.DMatrix(data = test_x, label = test_y)
  
  return(list(
    dtrain = dtrain,
    dtest = dtest,
    train_x = train_x,
    test_x = test_x,
    train_y = train_y,
    test_y = test_y
  ))
}


# Función para crear el modelo básico

xg_modelo_basico <- function(lista_datos, parametros, nrounds = 5000){
  modelo <- xgb.train(
    params = parametros,
    data = lista_datos$dtrain,
    nrounds = nrounds,
    watchlist = list(train = lista_datos$dtrain, test = lista_datos$dtest),
    print_every_n = 50
  )
  
  test_x <- lista_datos$test_x
  test_y <- lista_datos$test_y
  train_x <- lista_datos$train_x
  train_y <- lista_datos$train_y
  
  train_preds <- predict(modelo, train_x)
  test_preds<- predict(modelo, test_x)
  
  # RMSE (Root Mean Squared Error)
  
  rmse_train <- rmse(train_y, train_preds)
  rmse_test <- rmse(test_y, test_preds)
  
  # MAE (Mean Absolute Error)
  
  mae_train <- mae(train_y, train_preds)
  mae_test <- mae(test_y, test_preds)
  
  # R² (Coeficiente de determinación)
  
  r2_train <- cor(train_y, train_preds)^2
  r2_test <- cor(test_y, test_preds)^2
  
  # MAPE (Mean Absolute Porcentual Error)
  
  mape_train <- mape(train_y, train_preds)
  mape_test <- mape(test_y, test_preds)
  
  # NSE (Nash-Sutcliffe Efficiency)

  nse_train <- NSE(train_y, train_preds)
  nse_test <- NSE(test_y, test_preds)
    
  # Imprimir resultados
  
  cat("--METRICAS DE ENTRENAMIENTO--","\n",
      
      "RMSE:", rmse_train, "\n",
      "MAE:", mae_train, "\n",
      "MAPE:", mape_train, "\n",
      "R²:", r2_train, "\n",
      "NSE:", nse_train, "\n",
      "--METRICAS DE PRUEBA--","\n",
      "RMSE:", rmse_test, "\n",
      "MAE:", mae_test, "\n",
      "MAPE", mape_test, "\n",
      "R²:", r2_test, "\n",
      "NSE:", nse_test, "\n"
  )
  return(modelo)
}

# Función para entrenar y evaluar el modelo ajustado

xg_optimizar_modelo <- function(lista_datos, grid, cv = 5){
  control <- trainControl(
    method = "timeslice",
    initialWindow = floor(nrow(lista_datos$train_x) * 0.7),
    horizon = floor(nrow(lista_datos$train_x) * 0.3),
    fixedWindow = TRUE,
    verboseIter = FALSE
  )
  
  modelo <- train(
    x = lista_datos$train_x,
    y = lista_datos$train_y,
    method = "xgbTree",
    trControl = control,
    tuneGrid = grid
  )
  
  test_x <- lista_datos$test_x
  test_y <- lista_datos$test_y
  train_x <- lista_datos$train_x
  train_y <- lista_datos$train_y
  
  train_preds <- predict(modelo, lista_datos$train_x)
  test_preds<- predict(modelo, lista_datos$test_x)
  
  # RMSE (Root Mean Squared Error)
  
  rmse_train <- rmse(train_y, train_preds)
  rmse_test <- rmse(test_y, test_preds)
  
  # MAE (Mean Absolute Error)
  
  mae_train <- mae(train_y, train_preds)
  mae_test <- mae(test_y, test_preds)
  
  # R² (Coeficiente de determinación)
  
  r2_train <- cor(train_y, train_preds)^2
  r2_test <- cor(test_y, test_preds)^2
  
  # MAPE (Mean Absolute Porcentual Error)
  
  mape_train <- mape(train_y, train_preds)
  mape_test <- mape(test_y, test_preds)
  
  # NSE (Nash-Sutcliffe Efficiency)
  
  nse_train <- NSE(train_y, train_preds)
  nse_test <- NSE(test_y, test_preds)
  
  # Imprimir resultados
  
  cat("--METRICAS DE ENTRENAMIENTO--","\n",
      
      "RMSE:", rmse_train, "\n",
      "MAE:", mae_train, "\n",
      "MAPE:", mape_train, "\n",
      "R²:", r2_train, "\n",
      "NSE:", nse_train, "\n",
      "--METRICAS DE PRUEBA--","\n",
      "RMSE:", rmse_test, "\n",
      "MAE:", mae_test, "\n",
      "MAPE", mape_test, "\n",
      "R²:", r2_test, "\n",
      "NSE:", nse_test, "\n"
  )
  return(modelo)
}

# Función para visualizar resultados

xg_grafico_resultados <- function(modelo, lista_datos, title = "Valores reales vs Predicciones", opti = FALSE){
  if (opti){
    preds <- predict(modelo, lista_datos$test_x)
  }
  else{
    preds <- predict(modelo, lista_datos$dtest)
  }
  test_y <- lista_datos$test_y
  
  resultados <- data.frame(real = test_y, prediccion = preds)
  rmse_val <- round(rmse(test_y, preds), 2)
  
  ggplot(resultados, aes(x = real, y = prediccion)) +
    geom_point(color = "steelblue", alpha = 0.6, size = 3) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
    labs(
      title = title,
      x = "Valor Real",
      y = "Predicción"
    ) +
    annotate(
      "text", x = min(resultados$real), y = max(resultados$prediccion),
      label = paste("RMSE =", rmse_val),
      hjust = 0, vjust = 1, size = 5, color = "darkred"
    ) +
    theme_minimal()
}

# Función para visualizar residuos

xg_grafico_residuos <- function(modelo, lista_datos, opti = FALSE){
  if (opti){
    preds <- predict(modelo, lista_datos$test_x)
  }
  else{
    preds <- predict(modelo, lista_datos$dtest)
  }
  test_y <- lista_datos$test_y
  
  resultados <- data.frame(prediccion = preds, residual = test_y - preds)
  
  ggplot(resultados, aes(x = prediccion, y = residual)) +
    geom_point(color = "#D55E00", alpha = 0.7) +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
    labs(
      title = "Análisis de Residuales",
      x = "Predicción",
      y = "Residuos (Real - Predicción)"
    ) +
    theme_minimal()
}

# Función para visualizar importancia de las variables

xg_grafico_importancia <- function(modelo, top = 10, opti = FALSE){
  
  if (opti){
    importancia <- varImp(modelo)
    plot(importancia, top = top)
    
  }
  else{
    importancia <- xgb.importance(model = modelo)
    ggplot(importancia[1:top,], aes(x = reorder(Feature, Gain), y = Gain)) +
      geom_col(fill = "steelblue") +
      coord_flip() +
      labs(
        title = "Importancia de Variables",
        x = "Variable",
        y = "Importancia"
      ) +
      theme_minimal()
  }
}

# Función para visualizar series de tiempo

xg_serie_tiempo <- function(modelo, lista_datos, nombre_fecha, opti = FALSE){
  if (opti){
    preds <- predict(modelo, lista_datos$test_x)
  }
  else{
    preds <- predict(modelo, lista_datos$dtest)
  }
  
  temp <- as.data.frame(lista_datos$test_x)
  
  test_y <- lista_datos$test_y
  fechas <- as.Date(temp[[nombre_fecha]], origin = "1970-01-01")
  
  resultados <- data.frame(real = test_y, prediccion = preds, fecha = fechas)

  ggplot(resultados, aes(x = fecha)) + 
    geom_line(aes(y=real, color = "Valor Real"), size = 1.2)+
    geom_line(aes(y=prediccion,  color = "Predicción"), linetype = "dashed", size = 1.2) + 
    scale_color_manual(values = c("Valor Real" = "#1b9e77", "Predicción" = "#d95f02")) +
    scale_x_date(
      date_breaks = "6 months",
      date_labels = "%b %Y"
    ) +
    labs(
      title = "Comparación entre valores reales y predicciones",
      x = "Fecha",
      y = "Valor",
      color = "Leyenda"
    ) +
    theme_minimal(base_size = 12) + 
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      legend.position = "top",
      legend.title = element_text(face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
}

xg_histograma_residuos <- function(modelo, lista_datos, opti = FALSE){
  if (opti){
    preds <- predict(modelo, lista_datos$test_x)
  } else {
    preds <- predict(modelo, lista_datos$dtest)
  }
  test_y <- lista_datos$test_y
  residuos <- test_y - preds
  
  ggplot(data.frame(residuos = residuos), aes(x = residuos)) +
    geom_histogram(fill = "#56B4E9", color = "black", bins = 30) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
    labs(
      title = "Distribución de Residuos",
      x = "Residuo (Real - Predicción)",
      y = "Frecuencia"
    ) +
    theme_minimal()
}

xg_boxplot_residuos <- function(modelo, lista_datos, opti = FALSE){
  if (opti){
    preds <- predict(modelo, lista_datos$test_x)
  } else {
    preds <- predict(modelo, lista_datos$dtest)
  }
  test_y <- lista_datos$test_y
  residuos <- test_y - preds
  grupos <- cut(preds, breaks = quantile(preds, probs = seq(0, 1, 0.25)), include.lowest = TRUE)
  
  resultados <- data.frame(prediccion = preds, residuos = residuos, grupo = grupos)
  
  ggplot(resultados, aes(x = grupo, y = residuos)) +
    geom_boxplot(fill = "#F0E442") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    labs(
      title = "Boxplot de Residuos por Cuartil de Predicción",
      x = "Cuartil de Predicción",
      y = "Residuo"
    ) +
    theme_minimal()
}

xg_densidad_predicciones <- function(modelo, lista_datos, opti = FALSE){
  if (opti){
    preds <- predict(modelo, lista_datos$test_x)
  } else {
    preds <- predict(modelo, lista_datos$dtest)
  }
  test_y <- lista_datos$test_y
  
  resultados <- data.frame(
    valor = c(test_y, preds),
    tipo = rep(c("Real", "Predicción"), each = length(test_y))
  )
  
  ggplot(resultados, aes(x = valor, fill = tipo, color = tipo)) +
    geom_density(alpha = 0.4, size = 1.2) +
    labs(
      title = "Distribución de Valores Reales y Predicciones",
      x = "Valor",
      y = "Densidad",
      fill = "Tipo",
      color = "Tipo"
    ) +
    scale_fill_manual(values = c("Real" = "#1b9e77", "Predicción" = "#d95f02")) +
    scale_color_manual(values = c("Real" = "#1b9e77", "Predicción" = "#d95f02")) +
    theme_minimal()
}

xg_qqplot_residuos <- function(modelo, lista_datos, opti = FALSE){
  if (opti){
    preds <- predict(modelo, lista_datos$test_x)
  } else {
    preds <- predict(modelo, lista_datos$dtest)
  }
  test_y <- lista_datos$test_y
  residuos <- test_y - preds
  
  ggplot(data.frame(residuos = residuos), aes(sample = residuos)) +
    stat_qq(color = "#0072B2") +
    stat_qq_line(color = "red", linetype = "dashed") +
    labs(
      title = "QQ-Plot de Residuos",
      x = "Cuantiles Teóricos",
      y = "Cuantiles Observados"
    ) +
    theme_minimal()
}

xg_violin_residuos <- function(modelo, lista_datos, opti = FALSE){
  if (opti){
    preds <- predict(modelo, lista_datos$test_x)
  } else {
    preds <- predict(modelo, lista_datos$dtest)
  }
  test_y <- lista_datos$test_y
  residuos <- test_y - preds
  grupos <- cut(preds, breaks = quantile(preds, probs = seq(0, 1, 0.25)), include.lowest = TRUE)
  
  resultados <- data.frame(residuos = residuos, grupo = grupos)
  
  ggplot(resultados, aes(x = grupo, y = residuos)) +
    geom_violin(fill = "#009E73", alpha = 0.6) +
    geom_boxplot(width = 0.1, outlier.size = 0.5, fill = "white") +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
    labs(
      title = "Distribución de Residuos por Cuartil de Predicción",
      x = "Cuartil de Predicción",
      y = "Residuo"
    ) +
    theme_minimal()
}


### ----------------------------------- USO ------------------------------------- ###

# Predicción del caudal mínimo

## Brujo

### Modificar un poco la base
#base <- read.csv("data/Brujo/BaseBrujoModelo.csv")
#base <- base %>% 
#  select(-X)
#
#base <- base[!is.na(base[['caudal_minimo']]),]
#base$fecha <- as.Date(base$fecha)

### Generar las particiones de los datos

#lista <- xg_particion_datos(base, "caudal_minimo", 0.65, 123)

### Parámetros para el modelo

#parametros <- list(
#  objective = "reg:squarederror",
#  eval_metric = "rmse",
#  eta = 0.01,
#  max_depth = 10,
#  lambda = 2,
#  alpha = 0.1,
#  subsample = 0.7,
#  colsample_bytree = 0.7
#)

### Creación y evaluación del modelo

#modelo <- xg_modelo_basico(lista, parametros = parametros, nrounds = 5000)

### Graficar importancia de las variables

#xg_grafico_importancia(modelo, 15)

### Graficar comparación entre valores reales y predicciones

#xg_grafico_resultados(modelo, lista)

### Graficar residuos de las predicciones

#xg_grafico_residuos(modelo, lista)

### Graficar las series de tiempo

#xg_serie_tiempo(modelo = modelo, lista_datos = lista, nombre_fecha = "fecha")

### Graficar histograma del modelo

#xg_histograma_residuos(modelo, lista)

### Graficar boxplot del modelo

#xg_boxplot_residuos(modelo, lista)

### Graficar densidad del modelo

#xg_densidad_predicciones(modelo, lista)

### Gráficar QQ-Plot de residuos para el modelo

#xg_qqplot_residuos(modelo, lista)

### Graficar violin de residuos

#xg_violin_residuos(modelo, lista)

### Grid para optimización del modelo

#grid <- expand.grid(
#                    nrounds = 100,
#                    eta = c(0.01, 0.1, 0.3),
#                    max_depth = c(3, 6, 9),
#                    gamma = 0,
#                    colsample_bytree = 0.8,
#                    min_child_weight = 1,
#                    subsample = 0.8
#)

### Crear y evaluar modelo optimizado

#modelo_opti <- xg_optimizar_modelo(lista, grid)

### Graficar importancia de las variables del modelo optimizado

#xg_grafico_importancia(modelo_opti, 15, opti = TRUE)

### Graficar comparación entre valores reales y predicciones del modelo optimizado

#xg_grafico_resultados(modelo_opti, lista, opti = TRUE)

### Graficar residuos de las predicciones del modelo optimizado

#xg_grafico_residuos(modelo_opti, lista, opti = TRUE)

### Graficar serie de tiempo del modelo optimizado

#xg_serie_tiempo(modelo = modelo_opti, lista_datos = lista, nombre_fecha = "fecha", opti = TRUE)

### Graficar histograma del modelo ajustado

#xg_histograma_residuos(modelo_opti, lista, opti = TRUE)

### Graficar boxplot del modelo

#xg_boxplot_residuos(modelo_opti, lista, opti = TRUE)

