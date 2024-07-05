import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.getcwd)

# Importación de los datasets
contract_df = pd.read_csv('datasets/contract.csv')

def limpiar_columna(data):
    # Cambia las celdas que tienen ' ' por NaN
    data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)
    # Cambia el tipo de datos
    data['TotalCharges'] = data['TotalCharges'].astype(float)
    # Saca el promedio y remplaza los valores NaN con ese promedio
    promedio = round(data['TotalCharges'].mean(skipna=True), 2)
    data['TotalCharges'].fillna(promedio, inplace=True)
    return data

contract_df = limpiar_columna(contract_df)
print(contract_df.head(5))