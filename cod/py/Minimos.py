import pandas as pd

def cargar_csv(Cuenca: str, Base: str):
    '''
    Descarga un archivo CSV de la base de datos de la NASA y lo carga en un DataFrame de pandas.

    Parámetros
    ----------
        Cuenca : str
            Nombre de la cuenca para la que se desea cargar el CSV.
        Base : str
            Nombre de la base de datos que se desea cargar.
        
    Retorna
    -------
        pd.DataFrame
            DataFrame que contiene los datos del CSV.
    '''

    ruta = f'data/{Cuenca}/ICE/{Base}.csv'

    df = pd.read_csv(ruta)
    return df


def calcular_caudal_minimo_mensual(df):
    # Diccionario de conversión de meses en español a número
    meses_es = {'ENE': 1, 'FEB': 2, 'MAR': 3, 'ABR': 4, 'MAY': 5, 'JUN': 6,
                'JUL': 7, 'AGO': 8, 'SET': 9, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DIC': 12}

    # Paso 1: Convertir nombre del mes a número
    df['Mes_Num'] = df['Mes'].str.upper().map(meses_es)

    # Paso 2: Crear columna de fecha (manejo de errores)
    df['fecha'] = pd.to_datetime(
        dict(year=df['Ano'], month=df['Mes_Num'], day=df['Dia']),
        errors='coerce'
    )

    # Paso 3: Eliminar fechas inválidas
    df = df.dropna(subset=['fecha']).copy()

    # Paso 4: Crear columna de periodo mensual
    df['mes_periodo'] = df['fecha'].dt.to_period('M')

    # Paso 5: Índices de mínimos por mes
    idx_minimos = (
        df.groupby('mes_periodo')['Caudal_Diario']
        .idxmin()
        .dropna()
        .astype(int)
    )

    # Paso 6: Seleccionar filas de mínimos
    resultado = df.loc[idx_minimos].reset_index(drop=True)

    # Paso 7: Extraer año, mes en inglés, y día
    resultado['YEAR'] = resultado['fecha'].dt.year
    resultado['DIA'] = resultado['fecha'].dt.day
    resultado['MES'] = resultado['fecha'].dt.strftime('%b').str.upper()  # 'JAN', 'FEB', etc.

    # Paso 8: Reordenar columnas
    resultado = resultado[['YEAR', 'MES', 'DIA', 'Caudal_Diario']]
    resultado = resultado.rename(columns={'Caudal_Diario': 'caudal_minimo'})

    return resultado


#PRUEBA BRUJO
'''
df_brujo = cargar_csv('Brujo', 'BrujoDiarios(in)')

df_brujo['Caudal_Diario'] = pd.to_numeric(df_brujo['Caudal_Diario'], errors='coerce')

df_brujo = calcular_caudal_minimo_mensual(df_brujo)

df_brujo.to_csv('data/Brujo/ICE/BrujoMinimos.csv', index=False)
'''
#GUARDIA MINIMOS

df_guardia = cargar_csv('Guardia', 'GuardiaDiarios(in)')

df_guardia['Caudal_Diario'] = pd.to_numeric(df_guardia['Caudal_Diario'], errors='coerce')

df_guardia = calcular_caudal_minimo_mensual(df_guardia)

df_guardia.to_csv('data/Brujo/ICE/GuardiaMinimos.csv', index=False)
