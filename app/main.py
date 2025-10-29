from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count

# Crear sesión de Spark
spark = SparkSession.builder.appName("Exploracion_CSV").getOrCreate()

# === 1️⃣ Leer CSV ===
df = spark.read.csv("data/dataset.csv", header=True, inferSchema=True)

print("=== ESQUEMA DEL DATASET ===")
df.printSchema()

# === 2️⃣ Número total de filas ===
total_filas = df.count()
print(f"\nNúmero total de filas: {total_filas}")

# === 3️⃣ Conteo de nulos / vacíos por columna ===
print("\n=== DATOS FALTANTES POR COLUMNA ===")
missing = (
    df.select([
        count(when(col(c).isNull() | isnan(c) | (col(c) == ""), c)).alias(c)
        for c in df.columns
    ])
)
missing.show(truncate=False)

# === 4️⃣ Porcentaje de valores faltantes ===
print("=== PORCENTAJE DE FALTANTES POR COLUMNA ===")
missing_percent = (
    missing.withColumnRenamed(df.columns[0], "temp")
    .toPandas()
    .T
)
missing_percent.columns = ["faltantes"]
missing_percent["porcentaje"] = (missing_percent["faltantes"] / total_filas * 100).round(2)
print(missing_percent)

# === 5️⃣ Estadísticas descriptivas numéricas ===
print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
df.describe().show()

# === 6️⃣ Conteo de valores únicos por columna ===
print("\n=== VALORES ÚNICOS POR COLUMNA ===")
for c in df.columns:
    unicos = df.select(c).distinct().count()
    print(f"{c}: {unicos} valores únicos")

# Finalizar sesión
spark.stop()
