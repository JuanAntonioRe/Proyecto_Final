import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import os, sys
sys.path.append(os.getcwd)

# Verifica si la carpeta existe, si no la crea. Carpeta donde se guardará el csv
if not os.path.exists('files/datasets/output/'):
    os.makedirs('files/datasets/output/')

# Lee el dataset con codificación ohe
df = pd.read_csv("files/datasets/intermediate/a02_df_ohe.csv")

# Función para dividir el objetivo de las características
def features_target(data, column):
    features = data.drop(column, axis=1)
    target = data[column]
    return features, target

# Función para dividir en los conjuntos de validación y entrenamiento
def train_valid(features, target):
    fatures_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.2,
                                                                                 random_state=12345)
    return fatures_train, features_valid, target_train, target_valid

# Función para escalar características
def features_scaler(features_train, features_valid, numeric):
    scaler = StandardScaler()
    scaler.fit(features_train[numeric])
    features_train[numeric] = scaler.transform(features_train[numeric])
    features_valid[numeric] = scaler.transform(features_valid[numeric])
    
    return features_train, features_valid

# Función para sobremuestreo
def upsample(features, target, repeat=3):
    features_zeros = features[target==0]
    features_ones = features[target==1]
    target_zeros = target[target==0]
    target_ones = target[target==1]
    arg1 = pd.concat([features_zeros] + [features_ones] * repeat)
    arg2 = pd.concat([target_zeros] + [target_ones] * repeat)
    features_upsample, target_upsample = shuffle(arg1, arg2, random_state=12345)
    
    return features_upsample, target_upsample

# Función para submuestreo
def downsample(features, target, fraction=0.2):
    features_zeros = features[target==0]
    features_ones = features[target==1]
    target_zeros = target[target==0]
    target_ones = target[target==1]
    arg1 = pd.concat([features_zeros.sample(frac=fraction, random_state=12345)] + [features_ones])
    arg2 = pd.concat([target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones])
    features_downsample, target_downsample = shuffle(arg1, arg2, random_state=12345)
    
    return features_downsample, target_downsample

# División en features y target del dataset con codificación OHE en la columna 'ContractEnd'
features, target = features_target(df, 'contract_end')

# Set de entrenamiento y validación
features_train, features_valid, target_train, target_valid = train_valid(features, target)

# Llamada a la función features_scaler
features_train, features_valid = features_scaler(features_train, features_valid, ['monthly_charges', 'total_charges'])

# Llamamos a la función upsample en el conjunto de entrenamiento
features_train_up, target_train_up = upsample(features_train, target_train)

# Llamamos a la función downsample en el conjunto de entrenamiento
features_train_down, target_train_down = downsample(features_train, target_train)

print(features_train_up.shape)
print(target_train_up.shape)
print(features_train_down.shape)

# Guardamos los datasets
features_train_up.to_csv('files/datasets/output/a03_features_train_up.csv', index=False)
target_train_up.to_csv('files/datasets/output/a03_target_train_up.csv', index=False)
features_train_down.to_csv('files/datasets/output/a03_features_train_down.csv', index=False)
target_train_down.to_csv('files/datasets/output/a03_target_train_down.csv', index=False)
features_valid.to_csv('files/datasets/output/a03_features_valid.csv', index=False)
target_valid.to_csv('files/datasets/output/a03_target_valid.csv', index=False)

print("Se han guardado los datasets")