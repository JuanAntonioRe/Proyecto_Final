import pandas as pd
import re
import os, sys
sys.path.append(os.getcwd)

# Verifica si la carpeta existe, si no la crea. Carpeta donde se guardará el csv
if not os.path.exists('files/datasets/intermediate/'):
    os.makedirs('files/datasets/intermediate/')

df = pd.read_csv('files/datasets/intermediate/a01_merge_df_cleaned.csv')

# Función para convertir al estilo snake_case
def snake_case(name):
    string = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', string).lower()

# Función para convertir las columnas con 'Yes' y 'No' por unos y ceros
def yes_no_transform(df, columns):
    for column in columns:
        # Primero quitamos los valores None que pusimos
        df[column] = df[column].str.replace('Sin registro', df[column].mode().iloc[0])  # usamos iloc en dado caso que la columna tenga la misma cantidad de valores, sólo escogemos el primero
        
        # Cambiamos por 1 y ceros
        df[column] = df[column].map({'Yes': 1, 'No': 0, 'None':df[column].mode().iloc[0]})
        
        # También se cambia el tipo de datos
        df[column] = df[column].astype('int')

    return df

# Eliminación de columnas innecesarias
df = df.drop(['customerID', 'BeginDate', 'EndDate'], axis=1)

# Cambiamos a un estilo snake case
df = df.rename(columns=lambda x: snake_case(x))

# Lista de columnas a transformar
columns_to_transform = ['paperless_billing', 'partner', 'dependents','multiple_lines','online_security','online_backup',
                        'device_protection','tech_support','streaming_tv','streaming_movies'] 

#  Se llama a la función para transformar el dataset 'data'
df = yes_no_transform(df, columns_to_transform)

# Codificación OHE
df = pd.get_dummies(df, drop_first=True)

# Comprobamos la codificación
print(df.shape)

# Guardamos el dataset con codificación OHE
df.to_csv('files/datasets/intermediate/a02_df_ohe.csv', index=False)

