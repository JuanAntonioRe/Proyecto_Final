# Proyecto Final Bootcamp Data Science 🚀
Este proyecto representa el final del Bootcamp Data Science de TripleTen. Para su realización se debieron aplicar la mayoría de los conocimientos vistos durante el bootcamp.

## Objetivo del proyecto final
El objetivo es pronosticar la tasa de cancelación de clientes de la empresa Interconnect. De esta forma, si un usuario planea irse se le ofrecerán códigos promocionales y opciones de planes especiales.

Para poder realizar la tarea, el equipo de marketing de Interconnect ha recopilado algunos de los datos personales de sus clientes, incluyendo información sobre sus planes y contratos.

Para este preoyecto se ha establecido que la métrica clave es AUC-ROC.

Se tienen 4 datasets, los cuáles son:

- `contract.csv` — información del contrato;
- `personal.csv` — datos personales del cliente;
- `internet.csv` — información sobre los servicios de Internet;
- `phone.csv` — información sobre los servicios telefónicos.

## Desarrollo del proyecto
El proyecto tiene 4 etapas principales:
* Importación y preprocesamiento de los datos: En esta etapa se importan los datasets y se hace un preprocesamineto de datos buscando valores ausentes, duplicados, etc.
* Análisis Exploratorio de datos: Aquí se hace un análisis para diferentes preguntas realizadas. Hay gráficas (histogramas, boxplots) entregables que ayudan de manera visual a responder dichas preguntas.
* Modelos: Se entrenan 5 modelos diferentes (Desition Tree, Random Forest, Logistic Regression, LightGBM, XGBoost).
* Conclusiones: Se presentan las conclusiones.

## Alcance
Hay dos modelos que son capaces de predicir de manera efectiva si un cliente está a punto de irse de la empresa.

## Informes
El reporte detallado de todo el preyecto se encuentra en el archivo Eda.ipynb

## Tecnologías usadas
* Pandas
* Matplotlib
* Numpy
* Seaborn
* Scikit-learn
* LightGBM
* XGBoost
