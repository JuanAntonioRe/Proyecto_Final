import os, sys
sys.path.append(os.getcwd()) # Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código
import platform 

sistema_operativo = platform.system()

# Definir extension de ejecutables ---------------------------------------- 

if sistema_operativo == 'Windows':
        extension_binarios = ".exe"
else:
        extension_binarios = ""


# Preprocesos --------------------------------------------------------------

os.system(f"python{extension_binarios} preprocessing/a01_preproceso.py")
os.system(f"python{extension_binarios} preprocessing/a02_ohe.py")
os.system(f"python{extension_binarios} preprocessing/a03_train_split.py")


# Modelo ---------------------------------------------------------------------
os.system(f"python{extension_binarios} models/b01_creacion_de_modelos.py")