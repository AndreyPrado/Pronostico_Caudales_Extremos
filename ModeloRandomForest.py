import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from Modelo import Modelo as Modelo
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

class ModeloRandomForest(Modelo):
        
    def __init__(self, url, target_column, col_ignore):
        ''' Inicializa una instancia de ModeloRandomForest heredando de Modelo.

            Parámetros
            ----------
            url : str
                Ruta del archivo de datos a cargar
            target_column : str
                Nombre de la columna objetivo (variable dependiente)
            col_ignore : list
                Lista de columnas a ignorar en el modelo
                
            Retorna
            -------
            ModeloRandomForest
                Instancia de la clase inicializada
        '''
        super().__init__(url)
        self.__target_column = target_column
        self.__col_ignore = col_ignore
        self.__model = None
        self.__best = None

    #Getters y Setters 
    @property
    def modelo(self):
        ''' Obtiene el modelo RandomForest actual

            Parámetros
            ----------
            
            Retorna
            -------
            RandomForestRegressor
                Modelo de Random Forest almacenado
        '''
        return self.__modelo
    
    @modelo.setter
    def modelo(self, new):
        ''' Establece un nuevo modelo RandomForest
        
            Parámetros
            ----------
            new : RandomForestRegressor
                Nuevo modelo de Random Forest
                
            Retorna
            -------
            None
        '''
        self.__modelo = new
        
    def crear_y_ajustar_modelo(self):
        ''' Crea y ajusta un modelo RandomForest con búsqueda de hiperparámetros.

            Realiza una búsqueda en cuadrícula para encontrar los mejores hiperparámetros
            usando validación cruzada de series temporales.
            
            Parámetros
            ----------
            
            Retorna
            -------
            
        '''
        self.__model = RandomForestRegressor(random_state = 42)
        
        self._X_train, self._X_test, self._Y_train, self._Y_test = self.cargar_datos(
            x=self.__col_ignore, y=self.__target_column)
        
        grid = {
            'n_estimators': [100,300,500,700,1000],
            'max_depth': [10,15,20,25,30, None],
            'max_features': ['auto', 'sqrt', 'log2'],
            'min_samples_split': [2,5,10]
            }
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        grid_search = GridSearchCV(estimator = self.__model, param_grid = grid,
                                   cv=tscv, n_jobs=-1, scoring='neg_mean_squared_error',
                                   verbose=1)
        
        grid_search.fit(self._X_train, self._Y_train)
        
        print('Mejores parámetros', grid_search.best_params_)
        print('Mejor RMSE', np.sqrt(-grid_search.best_score_))
        
        self.__model = grid_search.best_estimator_
        

    def evaluar_modelo(self):
        ''' Evalúa el modelo RandomForest con múltiples métricas.
        
            Calcula R² para train/test, validación cruzada, MAE, MSE, RMSE,
            MAPE y NSE (Nash-Sutcliffe Efficiency).
            
            Parámetros
            ----------
            
            Retorna
            -------
            
        '''
        # Puntaje R2
        r2_train = self.__model.score(self._X_train, self._Y_train)
        r2_test = self.__model.score(self._X_test, self._Y_test)
    
        print(f"R² (train): {r2_train:.4f}")
        print(f"R² (test): {r2_test:.4f}")
    
        # Validación cruzada
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(self.__model, self._X_train, self._Y_train, cv=tscv, scoring = 'r2')
        print(f"R² promedio (CV): {cv_scores.mean():.4f}")
    
        # Predicciones
        self.__y_pred = self.__model.predict(self._X_test)
    
        # Métricas de error
        mae = mean_absolute_error(self._Y_test, self.__y_pred)
        mse = mean_squared_error(self._Y_test, self.__y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(self._Y_test, self.__y_pred) * 100  # porcentaje
    
        # NSE - Nash-Sutcliffe Efficiency
        y_obs = self._Y_test.values.ravel()
        y_pred = self.__y_pred.ravel()
        nse = 1 - np.sum((y_obs - y_pred) ** 2) / np.sum((y_obs - np.mean(y_obs)) ** 2)
    
        print(f"""
        Métricas de Error:
        - MAE (Error Absoluto Medio): {mae:.2f}
        - MSE (Error Cuadrático Medio): {mse:.2f}
        - RMSE (Raíz del Error Cuadrático Medio): {rmse:.2f}
        - MAPE (Error Porcentual Absoluto Medio): {mape:.2f}%
        - NSE (Eficiencia Nash–Sutcliffe): {nse:.4f}
        """)
        
    def graficar_predicciones(self):
        ''' Genera un gráfico comparando valores reales vs predicciones.

            Parámetros
            ----------
            
            Retorna
            -------
            matplotlib.figure.Figure
                Figura con el gráfico de comparación
        '''
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self._Y_test.values, label='Valores reales', linewidth=2)
        ax.plot(self.__y_pred, label='Predicciones', linewidth=2, linestyle='--')
        ax.set_title('Comparación de Predicciones vs Valores Reales')
        ax.set_xlabel('Observaciones')
        ax.set_ylabel('Valor')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        
        self.__grafico = fig
        
        return self.__grafico

    def visualizar_resultados(self):
        ''' Genera un gráfico de dispersión de valores reales vs predicciones.

            Parámetros
            ----------
            
            Retorna
            -------
            matplotlib.figure.Figure
                Figura con el gráfico de dispersión
        '''
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=self._Y_test.values.ravel(), y=self.__y_pred.ravel(), alpha=0.6, ax=ax)
        ax.plot([self._Y_test.min(), self._Y_test.max()], [self._Y_test.min(), self._Y_test.max()], 'r--')
        ax.set_title('Valores Reales vs Predicciones')
        ax.set_xlabel('Valores Reales')
        ax.set_ylabel('Predicciones')
        ax.grid(True)
        
        self.__grafico = fig
        
        return self.__grafico
    
        
    def importancia_feature(self, top_n = 10):
        ''' Calcula y visualiza la importancia de las variables según el modelo.

            Parámetros
            ----------
            top_n : int, opcional
                Número de variables más importantes a mostrar (por defecto 10)
                
            Retorna
            -------
            matplotlib.figure.Figure
                Figura con el gráfico de importancia de variables
        '''
        importances = self.__model.feature_importances_
        feature_names = self._X_train.columns
        df_impor = pd.DataFrame({
            'feature': feature_names,
            'importancia': importances
        }).sort_values(by='importancia', ascending=False)
    
        print(df_impor.head(top_n))
    
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='importancia', y='feature', data=df_impor.head(top_n), palette='viridis', ax=ax)
        ax.set_title('Importancia de las Variables del Modelo Random Forest')
        fig.tight_layout()
    
        self.__grafico = fig
        
        return self.__grafico
            
    def importancia_permutacion(self, nombre : str, top_n = 10 ):
        ''' Calcula y visualiza la importancia por permutación de las variables.
        
            Parámetros
            ----------
            nombre : str
                Tipo de importancia a visualizar ("media" o "sd")
            top_n : int, opcional
                Número de variables más importantes a mostrar (por defecto 10)
                
            Retorna
            -------
            matplotlib.figure.Figure
                Figura con el gráfico de importancia por permutación
        '''
        feature_names = self._X_train.columns
        resultados = permutation_importance(self.__model, self._X_test, self._Y_test, n_repeats=10, random_state=42)
        df_permu = pd.DataFrame({
            'feature': feature_names,
            'media_importancia': resultados.importances_mean,
            'sd_importancia': resultados.importances_std
        }).sort_values(by='media_importancia', ascending=False)
    
        print(df_permu.head(top_n))
        
        if nombre == "media":
            # Media
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='media_importancia', y='feature', data=df_permu.head(top_n).sort_values(by='media_importancia'), palette='viridis', ax=ax1)
            ax1.set_title('Importancia de las variables por Permutación (Media)')
            fig1.tight_layout()
            
            self.__grafico = fig1
        
        if nombre == "sd":
            # Desviación estándar
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='sd_importancia', y='feature', data=df_permu.head(top_n).sort_values(by='media_importancia'), palette='viridis', ax=ax2)
            ax2.set_title('Importancia de las variables por Permutación (Desviación Est.)')
            fig2.tight_layout()
            
            self.__grafico = fig2
            
        return self.__grafico
        
        