import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar el dataset
# Asegúrate de que el archivo 'dataset_limpio.csv' esté en la misma carpeta que este script
file_path = r"C:\Users\rodri\Documents\BIG DATA\TA BIG DATA\BIG-DATA---PROJECT\data\data_warehouse\dataset_limpio.csv\dataset_limpio.csv"
try:
    df = pd.read_csv(file_path)
    print("Dataset cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {file_path}")
    exit()

# 2. Preprocesamiento básico
# Seleccionamos solo las columnas numéricas, ya que la correlación de Pearson requiere números
df_numeric = df.select_dtypes(include=['float64', 'int64'])

# 3. Calcular la matriz de correlación
correlation_matrix = df_numeric.corr()

# 4. Enfocarse en la variable dependiente 'ingreso_total'
target_variable = 'ingreso_total'

if target_variable in correlation_matrix.columns:
    # Extraer la columna de correlaciones del target
    target_correlations = correlation_matrix[target_variable]
    
    # Eliminar la correlación con sí misma (que siempre es 1.0)
    target_correlations = target_correlations.drop(target_variable)
    
    # Ordenar de mayor a menor correlación (absoluta o directa)
    # Aquí ordenamos descendente para ver las relaciones positivas más fuertes arriba
    sorted_correlations = target_correlations.sort_values(ascending=False)
    
    print(f"\n--- Correlación de variables con '{target_variable}' ---")
    print(sorted_correlations)
    
    # 5. Visualización (Opcional pero recomendada)
    plt.figure(figsize=(12, 10))
    # Usamos un mapa de color 'coolwarm' donde rojo es positivo y azul es negativo
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f'Matriz de Correlación (Variable Objetivo: {target_variable})')
    plt.tight_layout()
    plt.show()
    
else:
    print(f"La variable '{target_variable}' no se encuentra en las columnas numéricas del dataset.")