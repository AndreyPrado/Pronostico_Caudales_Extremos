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
        self._datos = pd.read_excel(self._url).bfill()
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
        self._url = new_str
        
            
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
    def descargar_excel(self, nombre: str):
        ''' Método para descargar la base de datos en formato csv
        
        Parámetros
        ----------
        Nombre : str
            Corresponde al nombre con el que se desea guardar el archivo
        
        Retorna
        -------
        str
            String confirmando la descarga del excel    
        '''
        cadena = f"{nombre}.xlsx"
        self._datos.to_excel(cadena)
        return f"Archivo {nombre}.xlsx descargado con éxito"
        
    
