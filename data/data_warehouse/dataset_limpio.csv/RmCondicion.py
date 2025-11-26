import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# 1. Cargar datos
file_path = r"C:\Users\rodri\Documents\BIG DATA\TA BIG DATA\BIG-DATA---PROJECT\data\data_warehouse\dataset_limpio.csv\dataset_limpio.csv"
try:
    df = pd.read_csv(file_path)
    print("Dataset cargado correctamente.")
except FileNotFoundError:
    print("Error: Archivo no encontrado.")
    exit()

# --- 2. Selección de Variables (Feature Selection) ---

# Definimos la variable objetivo
target = 'condicion_ocupacion'

# Definimos las variables predictoras (Features)
# NOTA IMPORTANTE: Excluimos 'ingreso_total'. 
# ¿Por qué? Porque si alguien no tiene empleo, su ingreso suele ser NaN o 0. 
# Si el modelo ve eso, "hará trampa" y adivinará al 100% sin aprender patrones reales demográficos.
features = ['edad', 'sexo', 'nivel_educativo_cod', 'anios_educacion', 'conglomerado']

# Verificamos que las columnas existan antes de continuar
available_features = [col for col in features if col in df.columns]
print(f"Usando las siguientes variables para predecir: {available_features}")

X = df[available_features]
y = df[target]

# --- 3. Preprocesamiento (Limpieza) ---

# A) Manejo de Valores Nulos (Imputación)
# Random Forest en scikit-learn no acepta NaNs. Rellenaremos con la mediana.
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=available_features)

# B) Codificación del Target (Convertir texto a números)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Clases detectadas: {le.classes_}")

# --- 4. División del Dataset (Train / Test) ---
# Usamos 80% para entrenar y 20% para probar
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=0.2, random_state=42)

# --- 5. Creación y Entrenamiento del Modelo ---
print("\nEntrenando modelo Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --- 6. Predicciones y Evaluación ---
y_pred = rf_model.predict(X_test)

# Métricas
acc = accuracy_score(y_test, y_pred)
print(f"\n--- Resultados ---")
print(f"Precisión del modelo (Accuracy): {acc:.2%}")

print("\nReporte de Clasificación:")
# target_names nos permite ver las etiquetas originales en lugar de 0 y 1
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- 7. Visualización de Importancia de Variables ---
# Esto te dice qué variable pesó más en la decisión del modelo
feature_importances = pd.Series(rf_model.feature_importances_, index=available_features)
feature_importances = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns