import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l1_l2
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import shap
from sklearn.inspection import permutation_importance
from scipy.stats import randint as sp_randint
from tensorflow.keras.callbacks import CSVLogger, TensorBoard
from scipy.stats import uniform as sp_uniform
from Modelo import Modelo
from Grafico import Grafico


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
        self.__problema = 'regression'
        self.__mejores_params = None
        self.__tamaño_ventana = None

        self.__x_train_escalado = None
        self.__x_test_escalado = None
        self.__y_train_escalado = None
        self.__y_test_escalado = None
        self.__y_train = None
        self.__y_test = None
        
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
    
    @property
    def y_train(self):
        ''' Obtiene los valores de entrenamiento.
            
            Parámetros
            ----------
    
            Retorna
            -------
            numpy.ndarray
                Valores de entrenamiento
        '''
        return self.__y_train
    @property
    def y_test(self):
        ''' Obtiene los valores de prueba.
            
            Parámetros
            ----------
    
            Retorna
            -------
            numpy.ndarray
                Valores de prueba
        '''
        return self.__y_test
    @y_test.setter
    def y_test(self, value):
        ''' Establece los valores de prueba.
            
            Parámetros
            ----------
            value : numpy.ndarray
                Valores de prueba a establecer
        '''
        self.__y_test = value
    
    @y_train.setter
    def y_train(self, value):
        ''' Establece los valores de entrenamiento.
            
            Parámetros
            ----------
            value : numpy.ndarray
                Valores de entrenamiento a establecer
        '''
        self.__y_train = value

    
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
    def preprocesar_datos(self, variables_x, variable_y, tamaño_ventana: int, test_tam = 0.2):
        
        data_x = self._datos[variables_x].values
        data_y = self._datos[variable_y].values

        X, y = [], []
        for i in range(len(data_x) - tamaño_ventana):
            X.append(data_x[i:i + tamaño_ventana].flatten())
            y.append(data_y[i + tamaño_ventana])

        split_idx = int(len(X)*(1-test_tam))
        self.__y_train = np.array(y[:split_idx])
        self.__y_test = np.array(y[split_idx:])
        self.__x_train_escalado, self.__y_train_escalado = X[:split_idx], y[:split_idx]
        self.__x_test_escalado, self.__y_test_escalado = X[split_idx:], y[split_idx:]

        # Convertir listas a numpy arrays antes de escalar
        self.__x_train_escalado = self.__scaler_X.fit_transform(np.array(self.__x_train_escalado))
        self.__x_test_escalado = self.__scaler_X.transform(np.array(self.__x_test_escalado))

        self.__y_train_escalado = self.__scaler_y.fit_transform(np.array(self.__y_train_escalado).reshape(-1, 1)).flatten()
        self.__y_test_escalado = self.__scaler_y.transform(np.array(self.__y_test_escalado).reshape(-1, 1)).flatten()

        self.__variables_x = variables_x
        self.__tamaño_ventana = tamaño_ventana



        return self.__x_train_escalado, self.__x_test_escalado, self.__y_train_escalado, self.__y_test_escalado

    
    #Método para crear la arquitectura de la red neuronal
    def crea_modelo(self, hidden_layers=(64, 32), activation='relu', 
                     dropout_rate=0.2, learning_rate=0.001, 
                     regularization=0.01):
        
        model = Sequential()
        
        # Capa de entrada
        input_shape = (self.__x_train_escalado.shape[1],)
        model.add(Input(shape=input_shape))
        
        # Capas ocultas
        for units in hidden_layers:
            model.add(Dense(units, activation=activation,
                          kernel_regularizer=l1_l2(regularization)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Capa de salida
        
        model.add(Dense(1))
        loss = 'mean_squared_error'
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=['mae', 'mse'])

        self.__modelo = model
        
        return model
    
    #Optimizar Parámetros
    def opti_param(self, cv = 5, n_iter = 10):

        param_dist = {
            'hidden_layers': [(64,), (128, 64), (256, 128, 64)],
            'activation': ['relu', 'tanh', 'elu'],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.0001, 0.01],
            'batch_size': [16, 32, 64],
            'epochs': [100, 200],
            'regularization': [0.001, 0.01, 0.1]
        }

        model = KerasRegressor(build_fn=self.crea_modelo, verbose=0)
        cv_method = TimeSeriesSplit(n_splits=cv).split(self.__x_train_escalado)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv_method,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )
        
        search_result = search.fit(
            self.__x_train_escalado, 
            self.__y_train_escalado,
            callbacks=[EarlyStopping(patience=20, restore_best_weights=True)]
        )
        
        self.__mejores_params = search_result.best_params_
        print(f"Mejores parámetros: {self.__mejores_params}")
        
    #Método para entrenar el modelo
    def entrenar(self, epochs=1000, batch_size=32, verbose=0, validation_split=0.2):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6),
            TensorBoard(log_dir='./logs'),
            CSVLogger('training_log.csv')
        ]
        
        print("Entrenando el modelo...")
        self.__historial = self.__modelo.fit(
            self.__x_train_escalado, 
            self.__y_train_escalado,
            batch_size=batch_size,
            epochs=epochs, 
            verbose=verbose,
            validation_split=validation_split,
            callbacks=callbacks
        )
        print("Modelo entrenado!")
        
        return self.__historial
    
    def validacion_cruzada(self, n_splits=5, epochs=100, batch_size=32):
        """Implementa validación cruzada k-fold"""
        
        cv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_no = 1
        metrics = []
        
        for train_idx, val_idx in cv.split(self.__x_train_escalado):
            print(f"\nEntrenando fold {fold_no}...")
            
            # Dividir datos
            X_train_cv, X_val_cv = (self.__x_train_escalado[train_idx], 
                                    self.__x_train_escalado[val_idx])
            y_train_cv, y_val_cv = (self.__y_train_escalado[train_idx], 
                                   self.__y_train_escalado[val_idx])
            
            # Crear y entrenar modelo
            model = self.crea_modelo()
            history = model.fit(
                X_train_cv, y_train_cv,
                validation_data=(X_val_cv, y_val_cv),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[EarlyStopping(patience=20)]
            )
            
            # Evaluar
            score = model.evaluate(X_val_cv, y_val_cv, verbose=0)
            metrics.append(score)
            print(f"Fold {fold_no} - Loss: {score[0]:.4f} - MAE: {score[1]:.4f}")
            
            fold_no += 1
            
        # Calcular métricas promedio
        avg_metrics = np.mean(metrics, axis=0)
        print("\nMétricas promedio de validación cruzada:")
        print(f"Loss: {avg_metrics[0]:.4f} - MAE: {avg_metrics[1]:.4f}")
        
        return metrics
    
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
        rmse = np.sqrt(mean_squared_error(self.__y_test, self.__predicciones))
        mae = mean_absolute_error(self.__y_test, self.__predicciones)
        r2 = r2_score(self.__y_test, self.__predicciones)

    
        # Calcular NSE
        nse_numerador = np.sum((self.__y_test - self.__predicciones) ** 2)
        nse_denominador = np.sum((self.__y_test - np.mean(self.__y_test)) ** 2)
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

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.__y_test, label='Valores reales', linewidth=2)
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
        # Selección del background y conversión a DataFrame
        background = self.__x_train_escalado[np.random.choice(self.__x_train_escalado.shape[0], 100, replace=False)]
        df_background = pd.DataFrame(background, columns=self.__variables_x)
        df_datos = pd.DataFrame(self.__x_train_escalado, columns=self.__variables_x)

        # Explicador kernel (compatible con cualquier modelo)
        explainer = shap.KernelExplainer(self.__modelo.predict, df_background)

        # SHAP values
        shap_values = explainer.shap_values(df_datos, nsamples=100)

        # Gráfico
        shap.summary_plot(shap_values, df_datos, feature_names=self.__variables_x)
    #Método para calcular la importancia de las variables    
    def importancia_permu(self):

        model_sk = KerasRegressor(build_fn=lambda: self.__modelo, epochs=1, verbose=0)
        model_sk.fit(self.__x_train_escalado, self.__y_train_escalado, verbose=0)

        result = permutation_importance(
            model_sk,
            self.__x_train_escalado,
            self.__y_train_escalado,
            scoring="neg_mean_squared_error",
            n_repeats=10,
            random_state=42
        )

        sorted_idx = result.importances_mean.argsort()

        plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx])
        plt.yticks(range(len(sorted_idx)), np.array(self.__variables_x)[sorted_idx])
        plt.xlabel("Importancia por Permutación")
        plt.tight_layout()
        plt.show()

        
    def importancia_grad(self):

        x_tensor = tf.convert_to_tensor(self.__x_train_escalado, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            predicciones = self.__modelo(x_tensor, training=False)

        grad = tape.gradient(predicciones, x_tensor)

        if grad is None:
            raise ValueError("No se pudo calcular el gradiente. Revisa si el modelo es diferenciable o si usas funciones custom.")

        importancia = np.mean(np.abs(grad.numpy()), axis=0)

        plt.barh(range(len(importancia)), importancia)
        plt.yticks(range(len(importancia)), self.__variables_x)
        plt.xlabel('Importancia del Modelo con Gradientes')
        plt.tight_layout()
        plt.show()
