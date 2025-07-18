import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from BaseDatos import BaseDatos
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class Grafico(BaseDatos):
    
    def __init__(self, url):
        ''' Inicializa una instancia de la clase heredando la dirección de BaseDatos
            
            Parámetros
            ----------
            url : str
                Ruta del archivo csv
            
            Retorna
            -------
        '''
        super().__init__(url)
        self.__grafico = None
        
    #Método String
    def __str__(self):
        ''' Da una breve descripción de la clase
        
            Parámetros
            ----------
            
            Retorna
            -------
        '''
        return "Gráficos para la visualización de datos"
    

    #Método para graficar un boxplot    
    def boxplot(self, col = None):
        ''' Genera un gráfico de cajas para columnas numéricas.
        
            Parámetros
            ----------
            col : str o list
                Columna(s) a graficar. Si es None, usa todas las numéricas.
            
            Retorna
            -------
            fig : matplotlib.figure
                Figura con el gráfico generado
        '''
        df = self._datos
        
        if col is None:
            col = df.select_dtypes(include=["number"]).columns.tolist()
            
        fig, ax = plt.subplots(figsize=(10,5))
        sns.boxplot(data = df[col], ax=ax)
        ax.set_title("Gráfico de Cajas")
        plt.tight_layout()
        
        self.__grafico = fig
        
        return fig
    
    #Método para hacer un gráfico de líneas
    def linea(self, nombre_fechas :str, col :str):
        ''' Genera un gráfico de líneas para columnas numéricas.

            Parámetros
            ----------
            col : str o list
                Columna(s) a graficar. Si es None usa todas las numéricas.
            
            Retorna
            -------
            fig  : matplotlib.figure
                Figura con el gráfico generado.
        '''
        df = self._datos
        df[nombre_fechas] = pd.to_datetime(df[nombre_fechas], errors='coerce')
        
        años_unicos = []
        posicion = []
        actual = None
        
        for i, fecha in enumerate(df[nombre_fechas]):
            año = fecha.strftime("%Y")
            if año != actual:
                años_unicos.append(año)
                posicion.append(i)
                actual = año
            
        fig, ax = plt.subplots(figsize = (12,6))
        ax.plot(
            range(len(df)), 
            df[col],
            marker = "o",
            markersize = 2,
            markeredgewidth = 2,
            linestyle = "-",
            linewidth = 1,
            color = "b",
            label = col
        )
        if len(años_unicos) > 1:
            ax.set_xticks(posicion)
            ax.set_xticklabels(años_unicos, rotation = 45, ha="center")
        else:
            ax.tick_params(axis = 'x', rotation = 45)
    
        ax.set_xlabel("Fecha")
        ax.set_ylabel(f"{col}")
        ax.set_title(f"Serie de tiempo de la variable {col}")
        ax.grid(False)
        
        self.__grafico = fig
        
        return self.__grafico
    
    #Método para hacer un heatmap
    def heatmap(self):
        ''' Genera un mapa de calor de correlaciones entre variables numéricas.
        
            Parámetros
            ----------
            
            Retorna
            -------
            
                fig : matplot.figure
                    Figura con el heatmap de correlaciones
        '''
        
        df = self._datos.select_dtypes(include=["number"])
        corr = df.corr()
       
        fig, ax = plt.subplots(figsize=(20,20))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Mapa de Calor de Correlaciones")
        plt.tight_layout()
       
        self.__grafico = fig
       
        return fig
   
    #Método para graficas las distribuciones
    def dist(self, col = None):
        ''' Grafica la distribución de una columna con un histograma
        
            Parámetros
            ----------
            
            col : str o list
                Nombre(s) de la(s) columna(s) a graficar.
            
            Retorna
            -------
            
            fig : matplotlib.figure
                Figura con el gráfico de distribución
        '''
        df = self._datos
        
        if col is None:
            col = df.select_dtypes(include=["number"]).columns.tolist()

        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(df[col], kde = True, ax=ax)
        ax.set_title(f"Distribución de {col}")
        plt.tight_layout()
            
        self.__grafico = fig
            
        return fig
        
    #Método para guardar los gráficos en png
    def guardar_en_png(self, nombre: str):
        ''' Guarda el gráfico actual en un archivo PNG.
    
            Parámetros
            ----------
            
            nombre : str
                Nombre del archivo (sin extensión) donde se guardará el gráfico.
        
            Retorna
            -------
            
            str
                Mensaje indicando el resultado de la operación.
        '''
        if self.__grafico is None:
            return "No se ha generado ningún gráfico"
        else:
            self.__grafico.savefig(nombre+".png")
        
        return f"Gráfico guardado como {nombre}.png"
    
    def descomposicion(self, tipo : str, nombre : str):
        '''Descompone la serie en sus componentes (Residuos, Estacionalidad, Tendencia y la Original)

        Parámetros
        ----------
        tipo : str
            Tipo de descomposición ("STL", "aditive", "multiplicative")
        nombre : str
            Nombre de la columna que se quiere graficar

        Retorna
        -------
        descomposicion : matplotlib
            Resultado de la descomposición
        '''

        if tipo == "STL":
            #Se asume frecuencia mensual de la serie de tiempo
            stl = STL(self._datos[nombre], period=12, robust=True)
            descomposicion = stl.fit()

            #Genera los 4 gráficos
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
            
            ax1.plot(self._datos[nombre],label='Original', color='black')
            ax1.set_title("Original")
            
            ax2.plot(descomposicion.trend, label='Tendencia', color='blue')
            ax2.set_title("Tendencia")
            
            ax3.plot(descomposicion.seasonal, label='Estacionalidad', color='green')
            ax3.set_title("Estacionalidad")
            
            ax4.plot(descomposicion.resid, label='Residuos', color='red', linestyle='--')
            ax4.set_title("Residuos")
            
            plt.tight_layout()
            

        else:
            #Se asume frecuencia mensual de la serie de tiempo
            descomposicion = seasonal_decompose(self._datos[nombre], model = tipo, period = 12)
            descomposicion.plot()
          
        plt.show()
            
        return descomposicion
