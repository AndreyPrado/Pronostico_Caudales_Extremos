import pandas as pd

class BaseDatos():
    
    #Constructor de la Clase
    #Se asume que la base de datos ya está limpia
    def __init__(self, url: str):
        ''' Inicializa una instancia de la clase asumiendo que la base de datos ingresada
            ya está limpia y en formato tidy
            
            Parámetros
            ----------
            url : str
                Ruta del archivo csv de la base de datos
            
            Retorna
            -------
            
        '''
        self._url = url
        self._datos = pd.read_csv(self._url)
        #Se usará el método bfill para rellenar valores nulos
        self._tamano = self._datos.shape
    
    
    
    
    #Getters
    @property
    def url(self):
        ''' Devuelve la ruta actual del archivo
        
            Parámetros
            ----------
            
            Retorna
            -------
            
            str
                La ruta almacenada actual
        '''
        return self._url
    
    @property
    def datos(self):
        ''' Devuele la base de datos cargada
        
            Parámetros
            ----------
            
            Retorna
            -------
            
            pandas.DataFrame
                Base de datos
        '''
        return self._datos
    
    @property
    def tamano(self):
        ''' Devuelve las dimensiones de la base de datos
            
            Parámetros
            ----------
            
            Retorna
            -------
            
            Tupla con (filas, columnas) de la base de datos
        '''
        return self._tamano
    
    #Setters
    @url.setter
    def url(self, new_str : str):
        ''' Actualiza la ruta de los datos
        
            Parámetros
            ----------
            
            new_str : str
                Nueva ruta del archivo.
            
            Retorna
            -------
            
            '''
        self._datos = pd.DataFrame(new_str)
            
    @datos.setter
    def datos(self, new_df : pd.DataFrame):
        ''' Actualiza la base de datos
        
            Parámetros
            ----------
            
            new_df : pandas.DataFrame
                Nueva base de datos.
            
            Retorna
            -------
            
        '''
        self._datos = new_df
        self._tamano = self._datos.shape

    @tamano.setter
    def tamano(self, new_tamano : tuple):
        ''' Actualiza el tamaño de la base de datos
        
            Parámetros
            ----------
            
            new_tamano : tuple
                Nueva tupla con las dimensiones de la base de datos.
            
            Retorna
            -------
            
        '''
        if isinstance(new_tamano, tuple) and len(new_tamano) == 2:
            self._tamano = new_tamano
        else:
            raise ValueError("El tamaño debe ser una tupla con dos elementos (filas, columnas).")
        
    #Método String
    def __str__(self):
        ''' Da una descripción de la base de datos a utilizar
        Parámetros
        ----------
        
        Retorna
        -------
        
        '''
        return f"Base de Datos de dimensiones {self.tamano} \ny valores {self.datos}"
    
    #Método Para Descargar Pandas.DataFrame en CSV
    def descargar_csv(self, nombre: str):
        ''' Método para descargar la base de datos en formato csv
        Parámetros
        ----------
        Nombre : str
            Corresponde al nombre con el que se desea guardar el archivo
        
        Retorna
        -------
        
        '''
        cadena = nombre+".csv"
        self._datos.to_csv(cadena)
        return f"Archivo {nombre}.xlsx descargado con éxito"
    
    def filtrar(self):

        completos = self._datos.dropna()
        if completos.empty:
            raise ValueError("No hay filas completamente llenas en la base de datos.")
        idx_inicio = completos.index[0]
        idx_fin = completos.index[-1]
        # Filtra el DataFrame entre esos índices (inclusive)
        self._datos = self._datos.loc[idx_inicio:idx_fin].reset_index(drop=True)
        self._tamano = self._datos.shape
        return self._datos
         
    #Método para laguear las variables
    def lag_rango(self, cantidad: int, nombre: list):
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
        for col in nombre:
            for i in range(1, cantidad + 1):
                self._datos[f'{col}_lag_{i}'] = self._datos[col].shift(i) 
        
        return self._datos
    
    def lag_unico(self, cantidad, nombre):
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
        self._datos[f'{nombre}_lag_{cantidad}'] = self._datos[nombre].shift(cantidad)
        return self._datos
    
    def eliminar_columnas(self, columnas: list):
        ''' Elimina las columnas especificadas del DataFrame.
    
            Parámetros
            ----------
            columnas : list
                Lista de nombres de columnas a eliminar
            
            Retorna
            -------
            DataFrame
                DataFrame con las columnas eliminadas
        '''
        self._datos = self._datos.drop(columns=columnas)
        self._tamano = self._datos.shape
        return self._datos
    
    def eliminar_outliers(self, columna: str, umbral: float):
        ''' Elimina las filas donde los valores de la columna especificada son mayores que el umbral.
    
            Parámetros
            ----------
            columna : str
                Nombre de la columna a evaluar
            umbral : float
                Valor máximo permitido para esa columna
            
            Retorna
            -------
            DataFrame
                DataFrame con las filas eliminadas
        '''
        self._datos = self._datos[self._datos[columna] <= umbral].reset_index(drop=True)
        self._tamano = self._datos.shape
        return self._datos
    
    def fechas_num(self, nombre_fecha):

        self._datos['fecha'] = pd.to_datetime(self._datos[nombre_fecha], format='%Y-%m-%d', errors='coerce')
        self._datos['fecha']=(self._datos['fecha']-self._datos['fecha'].min()).dt.days