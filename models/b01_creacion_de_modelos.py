import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import time
import joblib
import os, sys
sys.path.append(os.getcwd)

# Verifica si la carpeta existe, si no la crea
if not os.path.exists('files/modeling_output'):
    os.makedirs('files/modeling_output')
    
if not os.path.exists('insight/models_plots'):
    os.makedirs('insight/models_plots')

# Leer los datos
features_train_up = pd.read_csv('files/datasets/output/a03_features_train_up.csv')
features_train_down = pd.read_csv('files/datasets/output/a03_features_train_down.csv')
target_train_up = pd.read_csv('files/datasets/output/a03_target_train_up.csv')
target_train_down = pd.read_csv('files/datasets/output/a03_target_train_down.csv')
features_valid = pd.read_csv('files/datasets/output/a03_features_valid.csv')
target_valid = pd.read_csv('files/datasets/output/a03_target_valid.csv')

# Función para entrenar y evaluar el mejor modelo
def eval_model(model, features_train, target_train, features_valid, target_valid, name):
    model.fit(features_train, target_train)
    predict_valid = model.predict(features_valid)
    predict_valid_prob = model.predict_proba(features_valid)[:,1]
    auc_roc = roc_auc_score(target_valid, predict_valid_prob)
    
    print(f'Modelo: {model.__class__.__name__}')
    print('Matriz de confusión:\n', confusion_matrix(target_valid, predict_valid))
    
    # Guarda el modelo
    joblib.dump(model, f'files/modeling_output/b01_{model.__class__.__name__}_{name}.joblib')
    
    name_plot = f'insight/models_plots/{model.__class__.__name__}_curva_roc_{name}.png'
        # Trazar la curva ROC
    fpr, tpr, _ = roc_curve(target_valid, predict_valid_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig(name_plot)
    plt.close()
    
    return auc_roc

# Función para seleccionar el mejor modelo
def grid_model(grid_search, features_train, target_train):
    grid_search.fit(features_train, target_train)
    
    best_model = grid_search.best_estimator_
    best_param = grid_search.best_params_
    return best_model, best_param

# Decision Tree ---------------------------------------------------------------------------------
# Hiperparámetros del modelo
param_tree = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 5, 10]
}

# Se crea el objeto GridSerachCV
grid_tree = GridSearchCV(estimator=DecisionTreeClassifier(random_state=12345), param_grid=param_tree, cv=2, scoring='roc_auc', verbose=False)

inicio = time.time()
# Buscamos el mejor modelo
best_model_tree_up, best_params = grid_model(grid_tree, features_train_up, target_train_up)
# Ya tenemos el mejor modelo, ahora calculamos el valor AUC-ROC
auc_roc = eval_model(best_model_tree_up, features_train_up, target_train_up, features_valid, target_valid, "sobremuestreo")
fin = time.time()
tiempo_modelo = fin - inicio
print('Valor AUC-ROC con datos de sobremuestro:', auc_roc)
print('Mejpres hiperparámetros:\n', best_params)
print(f'Tiempo del modelo: {tiempo_modelo} segundos')

inicio = time.time()
# Buscamos el mejor modelo
best_model_tree_down, best_params = grid_model(grid_tree, features_train_down, target_train_down)

# Ya tenemos el mejor modelo, ahora calculamos el valor AUC-ROC
auc_roc = eval_model(best_model_tree_down, features_train_down, target_train_down, features_valid, target_valid, "submuestreo")
fin = time.time()
tiempo_modelo = fin - inicio
print('Valor AUC-ROC con datos de submuestreo:', auc_roc)
print('Mejpres hiperparámetros:\n', best_params)
print(f'Tiempo del modelo: {tiempo_modelo} segundos')

# Random Forest ---------------------------------------------------------------------------------
# Hiperparámetros del modelo
param_rfc = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features' : ['auto', 'sqrt']
}

# Se crea el objeto GridSerachCV
grid_rfc = GridSearchCV(estimator=RandomForestClassifier(random_state=12345), param_grid=param_rfc, cv=2, scoring='roc_auc', verbose=False)

inicio = time.time()
# Buscamos el mejor modelo
best_model_rfc_up, best_params = grid_model(grid_rfc, features_train_up, target_train_up)
# Ya tenemos el mejor modelo, ahora calculamos el valor AUC-ROC
auc_roc = eval_model(best_model_rfc_up, features_train_up, target_train_up, features_valid, target_valid, "sobremuestreo")
fin = time.time()
tiempo_modelo = fin - inicio
print('Valor AUC-ROC con datos de sobremuestro:', auc_roc)
print('Mejpres hiperparámetros:\n', best_params)
print(f'Tiempo del modelo: {tiempo_modelo} segundos')

inicio = time.time()
# Buscamos el mejor modelo
best_model_rfc_down, best_params = grid_model(grid_rfc, features_train_down, target_train_down)

# Ya tenemos el mejor modelo, ahora calculamos el valor AUC-ROC
auc_roc = eval_model(best_model_rfc_down, features_train_down, target_train_down, features_valid, target_valid, "submuestreo")
fin = time.time()
tiempo_modelo = fin - inicio
print('Valor AUC-ROC con datos de submuestreo:', auc_roc)
print('Mejpres hiperparámetros:\n', best_params)
print(f'Tiempo del modelo: {tiempo_modelo} segundos')

# Logisitic Regression ---------------------------------------------------------------------------------
# Hiperparámetros del modelo
param_regression = {
    'C' : [0.1, 1, 5, 10],
    'solver':['lbfgs', 'liblinear', 'newton-cholesky'],
    'penalty': ['l1', 'l2', 'elasticnet']
    }

# Se crea el objeto GridSerachCV
grid_regression = GridSearchCV(estimator=LogisticRegression(random_state=12345), param_grid=param_regression, cv=2, scoring='roc_auc', verbose=False)

inicio = time.time()
# Buscamos el mejor modelo
best_model_regression_up, best_params = grid_model(grid_regression, features_train_up, target_train_up)
# Ya tenemos el mejor modelo, ahora calculamos el valor AUC-ROC
auc_roc = eval_model(best_model_regression_up, features_train_up, target_train_up, features_valid, target_valid, "sobremuestreo")
fin = time.time()
tiempo_modelo = fin - inicio
print('Valor AUC-ROC con datos de sobremuestro:', auc_roc)
print('Mejpres hiperparámetros:\n', best_params)
print(f'Tiempo del modelo: {tiempo_modelo} segundos')

inicio = time.time()
# Buscamos el mejor modelo
best_model_regression_down, best_params = grid_model(grid_regression, features_train_down, target_train_down)

# Ya tenemos el mejor modelo, ahora calculamos el valor AUC-ROC
auc_roc = eval_model(best_model_regression_down, features_train_down, target_train_down, features_valid, target_valid, "submuestreo")
fin = time.time()
tiempo_modelo = fin - inicio
print('Valor AUC-ROC con datos de submuestreo:', auc_roc)
print('Mejpres hiperparámetros:\n', best_params)
print(f'Tiempo del modelo: {tiempo_modelo} segundos')

# LightGBM ---------------------------------------------------------------------------------
# Hiperparámetros del modelo
param_lgbm = {
    'num_leaves': [31, 62, 127],
    'learning_rate':[0.01, 0.1, 0.2],
    'max_depth':[2, 5 , 8],
    'min_child_samples': [2, 5, 10, 20],
    'n_estimators':[100, 200, 300, 400, 500]
    }

# Se crea el objeto GridSerachCV
grid_lgbm = GridSearchCV(estimator=LGBMClassifier(random_state=12345), param_grid=param_lgbm, cv=2, scoring='roc_auc', verbose=False)

inicio = time.time()
# Buscamos el mejor modelo
best_model_lgbm_up, best_params = grid_model(grid_lgbm, features_train_up, target_train_up)
# Ya tenemos el mejor modelo, ahora calculamos el valor AUC-ROC
auc_roc = eval_model(best_model_lgbm_up, features_train_up, target_train_up, features_valid, target_valid, "sobremuestreo")
fin = time.time()
tiempo_modelo = fin - inicio
print('Valor AUC-ROC con datos de sobremuestro:', auc_roc)
print('Mejpres hiperparámetros:\n', best_params)
print(f'Tiempo del modelo: {tiempo_modelo} segundos')

inicio = time.time()
# Buscamos el mejor modelo
best_model_lgbm_down, best_params = grid_model(grid_lgbm, features_train_down, target_train_down)

# Ya tenemos el mejor modelo, ahora calculamos el valor AUC-ROC
auc_roc = eval_model(best_model_lgbm_down, features_train_down, target_train_down, features_valid, target_valid, "submuestreo")
fin = time.time()
tiempo_modelo = fin - inicio
print('Valor AUC-ROC con datos de submuestreo:', auc_roc)
print('Mejpres hiperparámetros:\n', best_params)
print(f'Tiempo del modelo: {tiempo_modelo} segundos')

# XGBoost ---------------------------------------------------------------------------------
# Hiperparámetros del modelo
param_xgb = {
    'learning_rate':[0.01, 0.1, 0.2],
    'max_depth':[2, 5, 6, 8],
    'n_estimators':[100, 200, 300, 400, 500],
    'alpha': [0, 0.1, 0.5, 1, 5],
    'lambda': [0, 0.1, 0.5, 1, 5]
    }

# Se crea el objeto GridSerachCV
grid_xgb = GridSearchCV(estimator=XGBClassifier(random_state=12345), param_grid=param_xgb, cv=2, scoring='roc_auc', verbose=False)

inicio = time.time()
# Buscamos el mejor modelo
best_model_xgb_up, best_params = grid_model(grid_xgb, features_train_up, target_train_up)
# Ya tenemos el mejor modelo, ahora calculamos el valor AUC-ROC
auc_roc = eval_model(best_model_xgb_up, features_train_up, target_train_up, features_valid, target_valid, "sobremuestreo")
fin = time.time()
tiempo_modelo = fin - inicio
print('Valor AUC-ROC con datos de sobremuestro:', auc_roc)
print('Mejpres hiperparámetros:\n', best_params)
print(f'Tiempo del modelo: {tiempo_modelo} segundos')

inicio = time.time()
# Buscamos el mejor modelo
best_model_xgb_down, best_params = grid_model(grid_xgb, features_train_down, target_train_down)

# Ya tenemos el mejor modelo, ahora calculamos el valor AUC-ROC
auc_roc = eval_model(best_model_xgb_down, features_train_down, target_train_down, features_valid, target_valid, "submuestreo")
fin = time.time()
tiempo_modelo = fin - inicio
print('Valor AUC-ROC con datos de submuestreo:', auc_roc)
print('Mejpres hiperparámetros:\n', best_params)
print(f'Tiempo del modelo: {tiempo_modelo} segundos')

print("----------------------------------------------")
print("Se ha terminado de evaluar a los modelos")