import pandas as pd

def cargar_csv(Cuenca: str, Base: str, Ruta = None):
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

    if Ruta is not None:
        ruta = Ruta
        df = pd.read_csv(ruta)
    else:
        ruta = f'data/{Cuenca}/NASA/{Base}.csv'

        with open(ruta, 'r') as file:

            for i, line in enumerate(file):

                if line.startswith('PARAMETER'):
                    filas_ignorar = i
                    break
        df = pd.read_csv(ruta, skiprows=filas_ignorar)
    
    return df

def leer_txt(Cuenca: str, Base: str):
    '''
    Carga un archivo de texto en un DataFrame de pandas.

    Parámetros
    ----------
        Cuenca : str
            Nombre de la cuenca para la que se desea cargar el archivo.
        Base : str
            Nombre del archivo que se desea cargar.
        
    Retorna
    -------
        pd.DataFrame
            DataFrame que contiene los datos del archivo de texto.
    '''

    ruta = f'data/{Cuenca}/ICE/{Base}.txt'
    
    df = pd.read_csv(ruta, sep='\\s+', usecols=['YR', 'MON', 'NINO3'])

    df = df.rename(columns={'YR': 'YEAR', 'MON': 'MES', 'NINO3': 'nino'})

    meses = {
        1: 'JAN',
        2: 'FEB',
        3: 'MAR',
        4: 'APR',
        5: 'MAY',
        6: 'JUN',
        7: 'JUL',
        8: 'AUG',
        9: 'SEP',
        10: 'OCT',
        11: 'NOV',
        12: 'DEC',
    }

    df['MES'] = df['MES'].map(meses)
    
    return df

def limpiar_base(df: pd.DataFrame, nombre: list):
    '''
    Limpia un DataFrame de pandas para prepararlo para su análisis.

    Parámetros
    ----------
        df : pd.DataFrame
            DataFrame que se desea limpiar.
        nombre : list
            Nombre de la columna que se añadirá al DataFrame limpio.
        
    Retorna
    -------
        pd.DataFrame
            DataFrame limpio y ordenado.
    '''

    df.columns = df.columns.str.strip() 

    unicas = df['PARAMETER'].unique()

    mapeo_nombres = dict(zip(unicas, nombre))
    
    lista_df = []

    for var in unicas:

        df_var = df[df['PARAMETER'] == var].copy()

        nombre_col = mapeo_nombres[var]

        df_tidy = df_var.melt(id_vars=['YEAR'], var_name='MES', value_name=nombre_col)

        df_tidy = df_tidy[df_tidy['MES'] != 'ANN']

        df_tidy[nombre_col] = pd.to_numeric(df_tidy[nombre_col], errors='coerce')

        lista_df.append(df_tidy)


    df_tidy = lista_df[0]

    for df in lista_df[1:]:
        df_tidy = pd.merge(df_tidy, df, on=['YEAR', 'MES'], how='outer')

    meses = {
        'JAN': 1, 'FEB': 2,'MAR': 3,'APR': 4,'MAY': 5,'JUN': 6,
        'JUL': 7,'AUG': 8,'SEP': 9,'OCT': 10,'NOV': 11,'DEC': 12}
    
    df_tidy['MES'] = pd.Categorical(df_tidy['MES'], categories=meses.keys(), ordered=True)

    df_tidy = df_tidy.sort_values(['YEAR', 'MES'])

    df_tidy = df_tidy.reset_index(drop=True)

    columnas = df_tidy.columns[2:]

    for col in columnas:
        df_tidy[col] = pd.to_numeric(df_tidy[col], errors='coerce')

    return df_tidy


def unir_bases(lista: list, cuenca: str, nombre : str):
    '''
    Une múltiples DataFrames de pandas en uno solo.

    Parámetros
    ----------
        lista : list
            Lista de DataFrames que se desea unir.
        nombre : str
            Nombre de la columna que se añadirá al DataFrame unido.
    '''

    df_unido = lista[0]
    for df in lista[1:]:
        df_unido = pd.merge(df_unido, df, on=['YEAR', 'MES'], how='outer')

    filtro = df_unido['MES'].notna() & (df_unido['MES'] != '')

    df_unido = df_unido[filtro]
    df_unido = df_unido.reset_index(drop=True)

    columnas = df_unido.columns[2:]

    for col in columnas:
        df_unido[col] = pd.to_numeric(df_unido[col], errors='coerce')

    meses_a_numero = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }

    df_unido['DIA'] = pd.to_numeric(df_unido['DIA'], errors='coerce').fillna(1).astype(int)

    df_unido['fecha'] = pd.to_datetime(
    df_unido['YEAR'].astype(str) + '-' + 
    df_unido['MES'].map(meses_a_numero).astype(str) + '-' + 
    df_unido['DIA'].astype(str),
    format='%Y-%m-%d',
    errors='coerce'
    )

    df_unido.drop(columns=['YEAR', 'MES', 'DIA'], inplace=True)

    df_unido.sort_values(by='fecha', inplace=True)

    df_unido.to_csv(f'data/{cuenca}/{nombre}.csv', index=False)

    
    
# BRUJO
'''
df_humedad = cargar_csv('Brujo', 'Humidity')
df_humedad = limpiar_base(df_humedad, ['humedad'])

df_temperatura = cargar_csv('Brujo', 'Temp2m')
df_temperatura = limpiar_base(df_temperatura, ['temp', 'temp_max', 'temp_min'])

df_precipitacion = cargar_csv('Brujo', 'Precipitation')
df_precipitacion = limpiar_base(df_precipitacion, ['prep'])

df_soil = cargar_csv('Brujo', 'Soil')
df_soil = limpiar_base(df_soil, ['soil_perfil', 'soil_superf'])

df_viento = cargar_csv('Brujo', 'Wind')
df_viento = limpiar_base(df_viento, ['dir_viento', 'vel_viento'])

df_nino = leer_txt('Brujo', 'nino123')

df_caudal = cargar_csv('Brujo', 'Caudal', 'data/Brujo/ICE/BrujoMinimos.csv')

lista = [df_humedad, df_temperatura, df_precipitacion, df_soil, df_viento, df_nino, df_caudal]

unir_bases(lista, 'Brujo', 'Brujo_NASA')
'''
# GUARDIA

df_humedad = cargar_csv('Guardia', 'Humidity')
df_humedad = limpiar_base(df_humedad, ['humedad'])

df_temperatura = cargar_csv('Guardia', 'Temp2m')
df_temperatura = limpiar_base(df_temperatura, ['temp', 'temp_max', 'temp_min'])

df_precipitacion = cargar_csv('Guardia', 'Precipitation')
df_precipitacion = limpiar_base(df_precipitacion, ['prep'])

df_soil = cargar_csv('Guardia', 'Soil')
df_soil = limpiar_base(df_soil, ['soil_perfil', 'soil_superf'])

df_viento = cargar_csv('Guardia', 'Wind')
df_viento = limpiar_base(df_viento, ['dir_viento', 'vel_viento'])

df_nino = leer_txt('Guardia', 'nino123')

df_caudal = cargar_csv('Guardia', 'Caudal', 'data/Guardia/ICE/GuardiaMinimos.csv')

lista = [df_humedad, df_temperatura, df_precipitacion, df_soil, df_viento, df_nino, df_caudal]

unir_bases(lista, 'Guardia', 'Guardia_NASA')



