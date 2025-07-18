library(tidyverse)

# Función para cargar la base

cargar_base <- function(ruta){
  base <- read.csv(ruta)
  return(base)
}

# Función para dar formato de fecha

formato_fechas <- function(base, nombre_fecha){
  base[[nombre_fecha]] <- as.Date(base[[nombre_fecha]])
  base <- base[order(base[[nombre_fecha]]), ]
  return(base)
}


# Función para hacer rezagos

rezagos <- function(base, columnas, num_lag, unico){
  if(unico == FALSE){
  for (col in columnas){
    for (i in 1:num_lag) {
      nombre <- paste0(col,"_lag_",i)
      base[[nombre]] <- lag(base[[col]], i)
    }
  }}
  else{
    nombre <- paste0(columnas,"_lag_",num_lag)
    base[[nombre]] <- lag(base[[columnas]], num_lag)
  }
  return(base)
}

eliminar <- function(base, col){
  base <- base[, !(names(base) %in% col)]
  return(base)
}

# Función para filtrar por fechas

filtrar <- function(base, fecha, fecha_inf, fecha_sup){
  base <- base %>% filter(fecha >= fecha_inf) %>% 
    filter(fecha <= fecha_sup)
  return(base)
}

# Función para guardar la base en csv

guardar <- function(base, nombre_cuenca){
  nombre <- paste0('data/',nombre_cuenca,'/Modelo',nombre_cuenca,'.csv')
  write.csv(base, nombre)
}

#base <- cargar_base("data/Guardia/Guardia_NASA.csv")
#base <- formato_fechas(base, "fecha")
#base <- rezagos(base, c("humedad","temp","temp_max","temp_min","prep","soil_perfil","soil_superf","dir_viento","vel_viento","nino","caudal_minimo"), 6, FALSE)
#base <- rezagos(base, "caudal_minimo",12, TRUE)
#base <- filtrar(base, fecha, as.Date("1981-04-29") ,as.Date("1993-03-16"))
#base <- eliminar(base, c("humedad","temp","temp_max","temp_min","prep","soil_perfil","soil_superf","dir_viento","vel_viento","nino"))
#guardar(base,'Guardia')




