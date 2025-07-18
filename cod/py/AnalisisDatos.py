import pandas as pd
from BaseDatos import BaseDatos

class AnalisisDatos(BaseDatos):
    
    #Constructor de la Clase AnalisisDatos
    def __init__(self, url):
        ''' Constructor que inicializa la clase heredada y calcula filas/columnas de los datos.

            Parámetros
            ----------
            url : str
                Ruta del archivo CSV con los datos.
            Retorna
            -------
            
        '''
        super().__init__(url)
        self.__filas = self._tamano[0]
        self.__columnas = self._tamano[1]
        
    #Getters
    @property
    def filas(self):
        ''' Devuelve la cantidad de filas de la base de datos
        
            Parámetros
            ----------
            
            Retorna
            -------
            int
                Cantidad de filas
        '''
        return self.__filas
    
    @property
    def columnas(self):
        ''' Devuelve la cantidad de columnas de la base de datos
         
             Parámetros
             ----------
             
             Retorna
             -------
             int
                 Cantidad de columnas
        '''       
        return self.__columnas
        
    #Método String    
    def __str__(self):
        ''' Da una descripción de la clase
        
            Parámetros
            ----------
            
            Retorna
            -------
            
        '''
        return "Base de Datos en Formato Pandas para aplicar métodos de análisis de datos"
    
    #Método para detectar Valores Nulos
    def nulos(self):
        ''' Detecta y resume los valores nulos en el DataFrame.
            
            Parámetros
            ----------

            Retorna
            -------
            resumen : dict
                Diccionario con columnas como índices y valores nulos como valores.
                Estructura: {columna: {"Índices": [lista], "Cantidad": int}}
        '''    
        df = self._datos
        resumen = {}
        
        for col in df.columns:
            indices = df[df[col].isnull()].index.tolist()
        
            if indices:
                resumen[col] = {
                    "Índices": indices,
                    "Cantidad": len(indices)}
        print(pd.DataFrame(resumen))
        return resumen
    
    #Método para Clasificar Columnas
    def est_basicas(self):
        ''' Calcula estadísticas básicas para columnas numéricas del DataFrame.
        
            Parámetros
            ----------
            
            Retorna
            -------
            resumen : dict
                Diccionario con {columna: {"min": float, "q1": float, "q2": float, 
                "q3": float, "max": float}} para cada columna numérica.
        '''
        cuantitativas = self._datos.select_dtypes(include=["number"]).columns.tolist()
        resumen = {}
        
        for col in cuantitativas:
            resumen[col] = {
                "min" :self._datos[col].describe()["min"],
                "q1" :self._datos[col].describe()["25%"],
                "q2" :self._datos[col].describe()["50%"],
                "q3" :self._datos[col].describe()["75%"],
                "max" :self._datos[col].describe()["max"]}
            
        return resumen
    
    #Método para Detectar Outliers
    
    def detectar_outliers(self):
        ''' Identifica outliers en columnas numéricas usando el método del rango intercuartílico.
            
            Parámetros
            ----------
            
            Retorna
            -------
            resumen : dict
                Diccionario con {columna: {"Índices": list, "Cantidad": int}} 
                para cada columna con outliers detectados.
        '''
        cuantitativas = self._datos.select_dtypes(include=["number"]).columns.tolist()
        est_bas = self.est_basicas()
        resumen = {}
        
        for col in cuantitativas:
            q1 = est_bas[col]["q1"]
            q3 = est_bas[col]["q3"]
            iqr = q3-q1
            liminf = q1-1.5*iqr
            limsup = q3+1.5*iqr
            val = self._datos[(self._datos[col]<liminf)|(self._datos[col]>limsup)]
            val = pd.DataFrame(val)
            
            if not val.empty:
                indices = val.index.to_list()
                resumen[col] = {
                    "Índices": indices,
                    "Cantidad": len(indices)}
        
        return resumen
        
