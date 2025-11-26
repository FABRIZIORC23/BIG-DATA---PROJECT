import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# --- 1. Cargar datos ---
# Asegúrate de que la ruta sea correcta. Si el script está junto al archivo, basta con el nombre.
file_path = r"C:\Users\rodri\Documents\BIG DATA\TA BIG DATA\BIG-DATA---PROJECT\data\data_warehouse\dataset_limpio.csv\dataset_limpio.csv"
try:
    df = pd.read_csv(file_path)
    print("Dataset cargado correctamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en {file_path}")
    exit()

# --- 2. Selección de Variables (Feature Selection) OPTIMIZADA ---
target = 'condicion_ocupacion'

# APLICAMOS EL VEREDICTO ANTERIOR:
# 1. Quitamos 'anios_educacion' (redundante con nivel_educativo).
# 2. Quitamos 'conglomerado' (es un ID geográfico, no predice comportamiento).
# 3. Quitamos 'estrato' (no tenía varianza).
features = ['edad', 'sexo', 'nivel_educativo_cod']

print(f"Variables predictoras seleccionadas: {features}")

X = df[features]
y = df[target]

# --- 3. Preprocesamiento ---

# A) Imputación (rellenar vacíos)
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)

# B) Codificación del Target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_ # Guardamos los nombres ("Desocupado", "Ocupado")
print(f"Clases detectadas: {class_names}")

# --- 4. División Train / Test ---
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=0.3, random_state=42)

# --- 5. Entrenamiento ---
print("\nEntrenando Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --- 6. Predicciones y Evaluación ---
y_pred = rf_model.predict(X_test)

# Métricas Numéricas
print(f"\n--- Resultados ---")
print(f"Precisión Global (Accuracy): {accuracy_score(y_test, y_pred):.2%}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=class_names))

# --- 7. VISUALIZACIÓN 1: Importancia de Variables ---
plt.figure(figsize=(10, 5))
feature_importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis')
plt.title('¿Qué variables pesaron más en la decisión?')
plt.xlabel('Importancia Relativa')
plt.tight_layout()
plt.show()

# --- 8. VISUALIZACIÓN 2: Matriz de Confusión (LO QUE PEDISTE) ---
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Matriz de Confusión: Realidad vs. Predicción')
plt.ylabel('Verdadero (Realidad)')
plt.xlabel('Predicho por el Modelo')
plt.tight_layout()
plt.show()