import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer

# 1. Cargar datos
file_path = r"C:\Users\rodri\Documents\BIG DATA\TA BIG DATA\BIG-DATA---PROJECT\data\data_warehouse\dataset_limpio.csv\dataset_limpio.csv"
try:
    df = pd.read_csv(file_path)
    print("Dataset cargado correctamente.")
except FileNotFoundError:
    print("Error: Archivo no encontrado.")
    exit()

# --- 2. Preprocesamiento Específico para Regresión ---

# PASO CRÍTICO: Eliminar filas donde no tenemos el dato objetivo (ingreso_total)
# No podemos entrenar al modelo con "incógnitas", necesitamos ejemplos con respuesta.
df_train = df.dropna(subset=['ingreso_total']).copy()
print(f"Registros originales: {len(df)}")
print(f"Registros útiles para entrenamiento (con ingreso conocido): {len(df_train)}")

# Selección de Variables Predictoras (Features)
# Excluimos 'condicion_ocupacion' porque si ya filtramos los que tienen ingreso,
# es muy probable que todos sean 'Ocupado', volviendo la variable inútil (constante).
features = ['edad', 'sexo', 'nivel_educativo_cod', 'anios_educacion']
target = 'ingreso_total'

X = df_train[features]
y = df_train[target]

# --- 3. Limpieza de Variables Predictoras ---
# Rellenar valores faltantes en las features (ej. edad o nivel educativo vacíos)
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)

# --- 4. División Train / Test ---
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# --- 5. Entrenamiento del Modelo ---
print("\nEntrenando RandomForestRegressor... (esto puede tardar un poco)")
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# --- 6. Predicciones y Evaluación ---
y_pred = rf_regressor.predict(X_test)

# Cálculo de Métricas
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Resultados del Modelo ---")
print(f"R² Score (Coeficiente de determinación): {r2:.4f}")
print(f"  -> (1.0 es perfecto, 0.0 es aleatorio. Valores > 0.3 suelen ser aceptables en ciencias sociales)")
print(f"Error Medio Absoluto (MAE): {mae:.2f}")
print(f"  -> (En promedio, el modelo se equivoca en esta cantidad de dinero)")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")

# --- 7. Visualización: Realidad vs Predicción ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
# Línea de referencia perfecta
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Ingreso Real')
plt.ylabel('Ingreso Predicho')
plt.title('Ingreso Real vs Predicho (La línea roja es la perfección)')
plt.show()

# --- 8. Importancia de Variables ---
feature_importances = pd.Series(rf_regressor.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='magma')
plt.title('¿Qué variables definen más el ingreso?')
plt.xlabel('Importancia Relativa')
plt.tight_layout()
plt.show()