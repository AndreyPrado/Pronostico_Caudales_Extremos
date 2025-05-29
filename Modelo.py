import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from Grafico import Grafico


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
        ''' Obtiene el conjunto X_train
        
            Parámetros
            ----------
            
            Retorna
            -------
            
            X_train : array
                Conjunto X_train
        '''
        return self._X_train
    
    @X_train.setter
    def X_train(self, value):
        ''' Guarda un nuevo conjunto x_train
        
            Parámetros
            ----------
            
            value : array
                Nuevos datos de entrenamiento
                
            Retorna
            -------
            
        '''
        self._X_train = value
    
    
    @property
    def X_test(self):
        ''' Obtiene el conjunto x_test
        
            Parámetros
            ----------
            
            Retorna
            -------
            
            self._x_test : array
                Conjunto x_test
        '''
            
        return self._X_test
    
    @X_test.setter
    def X_test(self, value):
        ''' Guarda un nuevo conjunto de prueba
    
            Parámetros
            ----------
            value: array
                Nuevos datos de prueba 
            
            Retorna
            -------
            
        '''
        self._X_test = value
    
    
    @property
    def Y_train(self):
        ''' Obtiene el conjunto de entrenamiento
        
            Parámetros
            ----------
    
            Retorna
            -------
            array
                Conjunto de entrenamiento
        '''
        return self._Y_train
    
    @Y_train.setter
    def Y_train(self, value):
        ''' Guarda un nuevo conjuntode entrenamiento
    
            Parámetros
            ----------
            value: array
                Nuevo conjunto de entrenamiento
            
            Retorna
            -------
            
        '''
        self._Y_train = value
    
    
    @property
    def Y_test(self):
        ''' Obtiene el conjunto de prueba
            
            Parámetros
            ----------
            
            Retorna
            -------
            array
                Obtiene el conjunto de prueba
        '''
        return self._Y_test
    
    @Y_test.setter
    def Y_test(self, value):
        ''' Guarda un nuevo conjunto de prueba
    
            Parámetros
            ----------
            value: array
                Nuevo conjunto de prueba
            
            Retorna
            -------
            
        '''
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
        X = self._datos.drop(x, axis = 1)
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
        
        self._datos = self._datos.bfill()
        
        return self._datos
    
    #Método para usar fechas
    def fechas(self, nombre_col : str):
        ''' Separa la variable de fecha como año y mes
        
            Parámetros
            ----------
            
            nombre_col : str
                Nombre de la columna con la fecha
                
            Retorna
            -------
            
        '''    
    
        self._datos[nombre_col] = pd.to_datetime(self._datos[nombre_col])
        self._datos['ano'] = self._datos[nombre_col].dt.year
        self._datos['mes'] = self._datos[nombre_col].dt.month
        self._datos = self._datos.drop(columns = [nombre_col])
        
    def eliminar_vari(self, nombre_col : str):
        ''' Método para eliminar una columna
        
            Parámetros
            ----------
            
            nombre_col : str
                Nombre de la columna que se desea eliminar
                
            Retorna
            -------
            
            self.__datos : pd.DataFrame
                Actualiza el set original de datos pero sin la variable escogida
        '''
        
        self._datos = self._datos.drop(columns = nombre_col)
        
        return self._datos
        
        
        