import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Modelo import Modelo
from Grafico import Grafico
import shap
from sklearn.inspection import permutation_importance
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger


class RedesNeuronales(Modelo, Grafico):
    
    #Constructor de la Clase
    def __init__(self, url):
        ''' Inicializa la clase con la ruta de los datos.
    
            Parámetros
            ----------
            url : str
                Ruta del archivo de datos
                
            Retorna
            -------
            
            '''
        super().__init__(url)
        self.__modelo = None
        self.__historial = None
        self.__predicciones = None
        self.__x_escalado = None
        self.__y_escalado = None
        self.__scaler_X = StandardScaler()
        self.__scaler_y = StandardScaler()
        self.__metricas = {}
        
    #Getters
    @property
    def modelo(self):
        ''' Obtiene el modelo de red neuronal.
        
            Parámetros
            ----------
    
            Retorna
            -------
            tf.keras.Model
                Modelo de red neuronal
        '''
        return self.__modelo
    
    @property
    def historial(self):
        ''' Obtiene el historial de entrenamiento del modelo.
            
            Parámetros
            ----------
            
            Retorna
            -------
            tf.keras.History
                Historial con métricas de entrenamiento
        '''
        return self.__historial
    
    @property
    def predicciones(self):
        ''' Obtiene las predicciones realizadas por el modelo.
            
            Parámetros
            ----------
        
            Retorna
            -------
            numpy.ndarray
                Array con las predicciones
        '''
        return self.__predicciones
    
    @property
    def metricas(self):
        ''' Obtiene las métricas de evaluación del modelo.
            
            Parámetros
            ----------
    
            Retorna
            -------
            dict
                Diccionario con las métricas calculadas
        '''
        return self.__metricas
    
    #Método String
    def __str__(self):
        ''' Representación en string de la clase.
            
            Parámetros
            ----------
            
            Retorna
            -------
            str
                Descripción textual del modelo
        '''
        return "Modelo de Red Neuronal para predicción"
    
    #Método para preprocesar datos
    def preprocesar_datos(self, variables_x, variable_y):
        ''' Preprocesa los datos escalándolos y preparándolos para el modelo.
        
            Parámetros
            ----------
            variables_x : list
                Lista de columnas a usar como variables independientes
            variable_y : str
                Nombre de la columna a usar como variable dependiente
            
            Retorna
            -------
            tupla
                Tupla con (X_escalado, y_escalado)
        '''

        
        
        self._X_train, self._X_test, self._Y_train, self._Y_test = self.cargar_datos(
            x=variables_x, y=variable_y)
        
        
        # Escalar los datos
        self.__x_train_escalado = self.__scaler_X.fit_transform(self._X_train)
        self.__x_test_escalado = self.__scaler_X.transform(self._X_test)
        self.__y_train_escalado = self.__scaler_y.fit_transform(self._Y_train.values.reshape(-1, 1))
        self.__y_test_escalado = self.__scaler_y.transform(self._Y_test.values.reshape(-1, 1))
        
        
        self.__variables_x =self._X_train.columns.tolist()
        
        return self.__x_train_escalado, self.__x_test_escalado, self.__y_train_escalado, self.__y_test_escalado, self._X_train, self._X_test, self._Y_train, self._Y_test
    
    #Método para crear la arquitectura de la red neuronal
    def crear_modelo(self, capas, learning_rate=0.001, funcion_perdida="mean_squared_error"):
        ''' Crea la arquitectura de la red neuronal.
    
            Parámetros
            ----------
            capas : list
                Lista de tuplas (neuronas, activacion) para cada capa
                
            learning_rate : float
                Tasa de aprendizaje (default 0.1)
                
            funcion_perdida : str
                Función de pérdida a usar (default "mean_squared_error")
            
            Retorna
            -------
            tf.keras.Model
                Modelo de red neuronal compilado
        '''
        
        # Construir modelo con las capas especificadas
        modelo = tf.keras.Sequential()
        input_shape = self.__x_train_escalado.shape[1]
        
        neuronas, activacion = capas[0]
        modelo.add(tf.keras.layers.Dense(units=neuronas, input_shape=(input_shape,), activation=activacion))
                
        # Resto de capas
        for neuronas, activacion in capas[1:]:
            modelo.add(tf.keras.layers.Dense(units=neuronas, activation=activacion))
            
        # Compilar el modelo
        modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),loss=funcion_perdida)
            
        self.__modelo = modelo
        return modelo
        
    #Método para entrenar el modelo
    def entrenar(self, epochs=1000, verbose=0, validation_split=0.2):
        '''Entrena el modelo de red neuronal.

            Parámetros
            ----------
                epochs : int
                    Número máximo de épocas de entrenamiento.
                verbose : int
                    Nivel de verbosidad durante el entrenamiento.
                validation_split : float
                    Porcentaje de datos para validación.
                
            Returns
            -------
                self.__historial : tf.keras.History
                    Objeto con historial de entrenamiento.
        '''
        callbacks = [
        EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=1e-7)
        ]
        print("Entrenando el modelo...")
        self.__historial = self.__modelo.fit(
            self.__x_train_escalado, 
            self.__y_train_escalado, 
            epochs=epochs, 
            verbose=verbose,
            validation_split=validation_split,
            callbacks=callbacks
        )
        print("Modelo entrenado!")
        
        return self.__historial
    
    #Método para graficar el historial de entrenamiento
    def graficar_perdidas(self):
        ''' Genera un gráfico de la evolución de la pérdida durante el entrenamiento.
            
            Parámetros
            ----------
            
            Retorna
            -------
            matplotlib.figure
                Figura con el gráfico generado
        '''
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.__historial.history['loss'], label='Pérdida de entrenamiento')
        
        if 'val_loss' in self.__historial.history:
            ax.plot(self.__historial.history['val_loss'], label='Pérdida de validación')
            
        ax.set_xlabel("Épocas")
        ax.set_ylabel("Pérdida")
        ax.set_title("Evolución de la pérdida durante el entrenamiento")
        ax.legend()
        ax.grid(True)
        
        self._Grafico__grafico = fig  # Acceso al atributo privado de la clase Grafico
        return fig
    
    #Método para hacer predicciones
    def predecir(self):
        ''' Realiza predicciones usando el modelo entrenado.
        
            Parámetros
            ----------
        
            Retorna
            -------
            numpy.ndarray
                Array con las predicciones
        '''
        # Hacer predicciones
        predicciones_escaladas = self.__modelo.predict(self.__x_test_escalado)
        
        # Revertir el escalado
        self.__predicciones = self.__scaler_y.inverse_transform(predicciones_escaladas).flatten()
        
        # Guardar también en el atributo de la clase padre
        self.prediccion = self.__predicciones
        
        return self.__predicciones
    
    #Método para evaluar el modelo
    def evaluar_modelo(self):
        ''' Evalúa el modelo calculando métricas de rendimiento.
    
            Parámetros
            ----------
            y_real : array
                Valores reales para comparar
    
            Retorna
            -------
            dict
                Diccionario con métricas (R², RMSE, MAE, NSE)
        '''
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(self._Y_test, self.__predicciones))
        mae = mean_absolute_error(self._Y_test, self.__predicciones)
        r2 = r2_score(self._Y_test, self.__predicciones)
    
        # Calcular NSE
        nse_numerador = np.sum((self._Y_test - self.__predicciones) ** 2)
        nse_denominador = np.sum((self._Y_test - np.mean(self._Y_test)) ** 2)
        nse = 1 - (nse_numerador / nse_denominador)
    
        # Guardar métricas
        self.__metricas = {
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae,
            'NSE': nse
        }
    
        # Imprimir métricas
        print(f"R²: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"NSE: {nse:.4f}")
    
        return self.__metricas
    
    #Método para graficar resultados
    def graficar_resultados(self, nombre_fecha):
        '''Genera gráfico comparando valores reales y predicciones.

            Parámetros
            ----------
                nombre_fecha : str
                    Nombre de la columna de la fecha
                
            Retorna
            -------
            
               self.__grafico : matplotlib.figure
                   Figura con la comparación.
        '''
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self._Y_test.values, label='Valores reales', linewidth=2)
        ax.plot(self.__predicciones, label='Predicciones', linewidth=2, linestyle='--')
        ax.set_title('Comparación de Predicciones vs Valores Reales')
        ax.set_xlabel('Observaciones')
        ax.set_ylabel('Valor')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        
        self.__grafico = fig
        
        return self.__grafico
    
    #Método para guardar el modelo
    def guardar_modelo(self, nombre):
        ''' Guarda el modelo entrenado en un archivo
    
            Parámetros
            ----------
            nombre : str
                Nombre del archivo
                
            Retorna
            -------
        '''
        if self.__modelo is None:
            raise ValueError("No hay modelo para guardar")
        
        self.__modelo.save(f"{nombre}.h5")
        print(f"Modelo guardado como {nombre}.h5")
    
    #Método para cargar un modelo guardado
    def cargar_modelo(self, ruta):
        ''' Carga un modelo previamente guardado.
    
            Parámetros
            ----------
            ruta : str
                Ruta del archivo modelo a cargar
            
            Retorna
            -------
            tf.keras.Model
                Modelo cargado
        '''
        self.__modelo = tf.keras.models.load_model(ruta)
        print(f"Modelo cargado desde {ruta}")
        return self.__modelo
    
    def importancia_shap(self):
        '''Calcula y muestra importancia de características con SHAP.
            
            Parámetros
            ----------
            
            Retorna
            -------
        '''
        explainer = shap.Explainer(self.__modelo, self.__x_train_escalado)
        shap_values = explainer(self.__x_train_escalado)
        shap.summary_plot(shap_values, features = self.__x_train_escalado, feature_names = self.__variables_x)
        
    def importancia_permu(self):
        '''Calcula y muestra importancia por permutación.
        
            Parámetros
            ----------
            
            Retorna
            -------
        '''
        model = KerasRegressor(build_fn = lambda: self.__modelo, epochs=0, verbose=0)
        model.fit(self.__x_train_escalado, self.__y_train_escalado)
        resultados = permutation_importance(model, self.__x_train_escalado, self.__y_train_escalado, n_repeats=10, random_state=42 )
        sorted_idx = resultados.importances_mean.argsort()
        plt.barh(range(len(sorted_idx)), resultados.importances_mean[sorted_idx])
        plt.yticks(range(len(sorted_idx)), np.array(self.__variables_x)[sorted_idx])
        plt.xlabel("Importancia del Modelo de Redes Neuronales con Permutación")
        plt.show()
        
    def importancia_grad(self):
        ''' Calcula y muestra la importancia de las variables por método del gradiente
        
            Parámetros
            ----------
            
            Retorna
            -------
        '''
        x_tensor = tf.convert_to_tensor(self.__x_train_escalado, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            predicciones = self.__modelo(x_tensor)
        grad = tape.gradient(predicciones, x_tensor)
        importancia = np.mean(np.abs(grad.numpy()), axis = 0)
        
        plt.barh(range(len(importancia)), importancia)
        plt.yticks(range(len(importancia)), self.__variables_x[:x_tensor.shape[1]])
        plt.xlabel('Importancia del Modelo de Redes Neuronales con Gradientes')
        plt.show()
        