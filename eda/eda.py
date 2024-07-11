import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
sys.path.append(os.getcwd)

# Verifica si la carpeta existe, si no la crea
if not os.path.exists('insight/plots'):
    os.makedirs('insight/plots')

# Leemos el dataset limpio
df = pd.read_csv('files/datasets/intermediate/a01_merge_df_cleaned')

# Función para plot para ver los clientes con contrato vigente
def clientes_contrato_vigente(df):
    fig, ax = plt.subplots(figsize=(8,8))

    dft = df['ContractEnd'].map({0:'No', 1:'Si'}).value_counts()
    dft.plot(kind='bar', ax=ax, color=['darkblue', 'orange'])
    ax.set_title('Clientes con contrato vigente')
    ax.set_xlabel('¿Terminó su contrato?')
    ax.set_ylabel('Cantidad de clientes')

    # Coloca las etiquetas a las barras
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2, p.get_height(), f'{p.get_height():.0f}', ha='center', va='bottom')
    
    # Rota las etiquetas del eje x
    for tick in ax.get_xticklabels():
        tick.set_rotation(360)
    
    # Guarda la figura
    plt.savefig(r'insight/plots/clientes_contrato_vigente.png')

# Función para plot de duración de los contratos
def duracion_contrato(df):
    # Creamos un df que contiene sólo las columnas de 'BeginDate' y 'EndDate' de las personas que ya no tienen contrato
    duration_df = df[df['ContractEnd'] == 1][['BeginDate', 'EndDate']]
    
    # Ahora cambiamos el tipo de objeto de las fechas
    duration_df['BeginDate'] = pd.to_datetime(duration_df['BeginDate'])
    duration_df['EndDate'] = pd.to_datetime(duration_df['EndDate'])

    # Calculamos los días meses de duaración de los contratos
    duration_df['Duration'] = round((duration_df['EndDate'] - duration_df['BeginDate']).dt.days / 30, 1)
    # Grafiquemos los meses que duró el contrato
    fig, ax = plt.subplots(figsize=(12,8), layout='constrained')

    dft = duration_df.groupby('Duration')['Duration'].value_counts()
    dft.plot(kind='bar', ax=ax)
    ax.set_title('Duración del contrato (meses)')
    ax.set_xlabel('Meses de duración')
    ax.set_ylabel('Cantidad de clientes')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
        
    plt.savefig(r'insight/plots/duracion_contrato.png')
    
def cargos_mensuales(df):
    # Eliminamos la columna 'TotalCharges'
    dft = df.drop(['TotalCharges'], axis=1)
    # Boxplot para cada tipo de cliente
    plt.figure(figsize=(8,6))
    sns.boxplot(x='MonthlyCharges', hue='ContractEnd', data=dft)
    plt.title('Cargos mensuales')
    plt.xlabel('Cargos mensuales (USD)')
    plt.legend(['Con contrato', 'Sin contrato'])
    plt.savefig(r'insight/plots/cargos_mensuales.png')

def pareja_hijos(df):
    dft = df[df['ContractEnd'] == 1].groupby(['Partner', 'Dependents'])['Dependents'].value_counts()
    dft = dft.reset_index()

    sns.barplot(x=dft['Partner'], y=dft['count'], hue=dft['Dependents'])
    plt.title("Conteo de hijos  de los clientes sin contrato")
    plt.xlabel("Pareja")
    plt.ylabel("Cantidad")
    plt.legend(title="Dependientes")
    plt.savefig(r'insight/plots/pareja_hijos_plot.png')

def tipo_de_contrato(df):
    # Gráfica de los clientes que ya no tienen contrato
    # Filtramos el DF
    dft = df[df['ContractEnd'] == 1].groupby('Type')['Type'].value_counts()

    # Graficamos
    fig, ax = plt.subplots(figsize=(6,8), layout='constrained')
    bars = dft.plot(kind='bar', ax=ax, title='Cantidad del tipo de contrato (clientes sin contrato)')
    ax.set_xlabel('Tipo de contrato')
    ax.set_ylabel('Cantidad de clientes')
    for p in bars.patches:
        ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2, p.get_height()), 
                    ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    plt.savefig(r'insight/plots/tipo_contrato_clientes_sin_contrato.png')
    
    # Gráfica de los clientes que siguen con contrato
    # Filtramos el DF
    dft2 = df[df['ContractEnd'] == 0].groupby('Type')['Type'].value_counts()

    # Graficamos
    fig, ax = plt.subplots(figsize=(6,8), layout='constrained')
    bars = dft2.plot(kind='bar', ax=ax, title='Cantidad del tipo de contrato (clientes con contrato)')
    ax.set_xlabel('Tipo de contrato')
    ax.set_ylabel('Cantidad de clientes')
    for p in bars.patches:
        ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2, p.get_height()), 
                    ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    plt.savefig(r'insight/plots/tipo_contrato_clientes_con_contrato.png')
        
clientes_contrato_vigente(df)
duracion_contrato(df)
cargos_mensuales(df)
pareja_hijos(df)
tipo_de_contrato(df)