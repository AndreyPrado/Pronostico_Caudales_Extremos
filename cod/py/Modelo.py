import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from Grafico import Grafico
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression


class Modelo(Grafico):
    
    #Constructor de la Clase
    def __init__(self, url):
        ''' Inicializa la instancia de Modelo heredando de BaseDatos.

            Parámetros
            ----------
            url : str
                Ruta del archivo de datos a cargar.
                
            Retorna
            -------
            
        '''
        super().__init__(url)
        self.__prediccion = None
        
    @property
    def prediccion(self):
        ''' Devuelve los resultados de la predicción del modelo
        
            Parámetros
            ----------
            
            Retorna
            -------
            
            pandas.DataFrame
                Resultados de la predicción
        '''
        return self.__prediccion
    
    @prediccion.setter
    def prediccion(self, new_value):
        ''' Guarda un nuevo valor para el atributo prediccion
            
            Parámetros
            ----------
            new_value: array
                Nuevas predicciones asignadas
            
            Retorna
            -------
            
        '''
        self.__prediccion = new_value
        
    @property
    def X_train(self):
        return self._X_train
    
    @X_train.setter
    def X_train(self, value):
        self._X_train = value
    
    
    @property
    def X_test(self):
        return self._X_test
    
    @X_test.setter
    def X_test(self, value):
        self._X_test = value
    
    
    @property
    def Y_train(self):
        return self._Y_train
    
    @Y_train.setter
    def Y_train(self, value):
        self._Y_train = value
    
    
    @property
    def Y_test(self):
        return self._Y_test
    
    @Y_test.setter
    def Y_test(self, value):
        self._Y_test = value
    #Método String
    def __str__(self):
        ''' Da una breve descripción de la clase
            
            Parámetros
            ----------
            
            Retorna
            -------
            str
                Breve Descripción de la clase
        '''
        
        return "Objeto Tipo Modelo para cargar los datos y separarlos."
    
    def imputar_datos_faltantes(self, freq: int = 12, model: str = 'additive'):
        columnas = [col for col in self._datos.columns if col != 'fecha']  # Excluye columna fecha si existe
        
        for col in columnas:
            serie = self._datos[col].copy()
            mascara = serie.isna()
            
            # Paso 1: Interpolación avanzada
            serie_inter = serie.interpolate(
                method='linear',
                limit_direction='both',
                limit_area='inside'  # Solo rellena entre valores válidos
            )
            
            # Paso 2: Relleno agresivo para NaN residuales
            # Primero hacia adelante, luego hacia atrás
            serie_inter = serie_inter.ffill().bfill()
            
            # Paso 3: Si aún hay NaN (columnas completamente vacías)
            if serie_inter.isna().any():
                # Usar la media/mediana como último recurso
                fill_value = serie_inter.mean() if pd.api.types.is_numeric_dtype(serie_inter) else serie_inter.mode()[0]
                serie_inter = serie_inter.fillna(fill_value)
            
            # Validación crítica
            if serie_inter.isna().any():
                raise ValueError(f"¡Error crítico! No se pudieron imputar todos los valores en {col}")
            
            # Paso 4: Para series temporales, aplicar descomposición estacional
            if pd.api.types.is_numeric_dtype(serie_inter):
                try:
                    decomposition = seasonal_decompose(
                        serie_inter,
                        period=freq,
                        model=model,
                        extrapolate_trend='freq'  # Manejo de bordes
                    )
                    serie_rec = decomposition.trend + decomposition.seasonal
                    self._datos[col] = np.where(mascara, serie_rec, serie)
                except:
                    # Fallback a interpolación si falla la descomposición
                    self._datos[col] = serie_inter
            else:
                # Para datos no numéricos
                self._datos[col] = serie_inter

    #Método para cargar las columnas
    def cargar_datos(self, x: list, y:list):
        ''' Divide los datos en conjuntos de entrenamiento y prueba usando TimeSeriesSplit.
    
            Parámetros
            ----------
            x : list
                Lista de columnas a excluir de X 
            y : list
                Lista de columnas a incluir en Y 
            
            Retorna
            -------
            X_train, X_test, Y_train, Y_test
                Tuplas con los conjuntos de entrenamiento y prueba
        '''
        X = self._datos.drop(columns = x, axis = 1)
        Y = self._datos[y]
        
        tss = TimeSeriesSplit(n_splits = 5)
        for train_index, test_index in tss.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
         
        return X_train, X_test, Y_train, Y_test
    
    #Método para hallar la correlación más alta
    def top_corr(self, percentil):
        ''' Encuentra y muestra las correlaciones más altas entre variables numéricas.
    
            Parámetros
            ----------
            percentil : int
                Percentil para establecer el umbral de correlación (entre 0 y 1)
            
            Retorna
            -------
        '''
        matriz = self._datos.select_dtypes(include = "number").corr()
        valores = matriz.values
        columns = matriz.columns
        index = matriz.index
        
        np.fill_diagonal(valores, np.nan)
        threshold = np.nanpercentile(valores, percentil)
        high_corr = np.argwhere(valores >= threshold)
        
        high_corr = high_corr[high_corr[:,0]<high_corr[:,1]]
        
        if high_corr.size >0:

            sorted_indices = np.argsort(-valores[high_corr[:,0], high_corr[:,1]])
            high_corr = high_corr[sorted_indices]
        
        print(f"Correlaciones >= {threshold:.2f} (percentil {percentil}):")
        for row, col in high_corr:
            print(f"{columns[row]} vs {index[col]} -> {valores[row,col]:.2f}")
        
        
    #Método para laguear las variables
    def lag(self, cantidad: int, nombre: str):
        ''' Crea una nueva columna con valores desplazados (lag) de la columna especificada.
    
            Parámetros
            ----------
            cantidad : int
                Número de períodos para el desplazamiento
            nombre : str
                Nombre de la columna a la que se aplicará el lag
            
            Retorna
            -------
            DataFrame
                DataFrame con la nueva columna lag añadida
        '''
        nombre_final = nombre+"_lag_"+str(cantidad)
        self._datos[nombre_final] = self._datos[nombre].shift(cantidad)
        
        return self._datos
    
    #Método para usar fechas
    def fechas(self, nombre_col : str):
        
        self._datos[nombre_col] = pd.to_datetime(self._datos[nombre_col])
        self._datos['ano'] = self._datos[nombre_col].dt.year
        self._datos['mes'] = self._datos[nombre_col].dt.month
        self._datos = self._datos.drop(columns = [nombre_col])
        
    def eliminar_vari(self, nombre_col : str):
        
        self._datos = self._datos.drop(columns = nombre_col)
        
        return self._datos
    
    def filtro_fila(self, numero:int):

        self._datos = self._datos.iloc[numero:]
        
        
        