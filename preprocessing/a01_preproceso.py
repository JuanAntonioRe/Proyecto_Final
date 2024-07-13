import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.getcwd)

# Verifica si la carpeta existe, si no la crea. Carpeta donde se guardará el csv
if not os.path.exists('files/datasets/intermediate/'):
    os.makedirs('files/datasets/intermediate/')

# Importación de los datasets
contract_df = pd.read_csv('files/datasets/input/contract.csv')
personal_df = pd.read_csv('files/datasets/input/personal.csv')
internet_df = pd.read_csv('files/datasets/input/internet.csv')
phone_df = pd.read_csv('files/datasets/input/phone.csv')

def limpiar_columna(data):
    # Cambia las celdas que tienen ' ' por NaN
    data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)
    # Cambia el tipo de datos
    data['TotalCharges'] = data['TotalCharges'].astype(float)
    # Saca el promedio y remplaza los valores NaN con ese promedio
    promedio = round(data['TotalCharges'].mean(skipna=True), 2)
    data['TotalCharges'].fillna(promedio, inplace=True)
    return data

# Función que coloca cero o 1 en la nueva columna si se ha terminado el contrato
def ContractEnding(end_date):
    if end_date == 'No':
        return 0
    else:
        return 1
    
contract_df = limpiar_columna(contract_df)
contract_df['ContractEnd'] = contract_df['EndDate'].apply(ContractEnding)

# Juntando todos los datasets en uno solo
merge_df = contract_df.merge(personal_df, on='customerID', how='left').merge(internet_df, on='customerID', how='left').merge(phone_df, on='customerID', how='left')

# Checando valores NaN
merge_df.info()
# Se quitan los valores NaN
merge_df = merge_df.fillna('Sin registro')
merge_df.info()

merge_df.to_csv('files/datasets/intermediate/a01_merge_df_cleaned.csv')

print()
print('Se ha guardado el csv')