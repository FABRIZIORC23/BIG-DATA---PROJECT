from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# --- CONFIGURACIÓN Y SESIÓN ---
spark = SparkSession.builder.appName("ETL: Extraccion, Transformacion y Carga (DW)").getOrCreate()
RUTAS_BASE = "/opt/spark-data/data_warehouse/"
SOURCE_CSV_PATH = "/opt/spark-data/dataset.csv"

# ----------------------------------------------------------------------------------
# FASE 1: EXTRACCIÓN Y LIMPIEZA INICIAL (E)
# ----------------------------------------------------------------------------------
print("\n FASE 1: Extracción y Limpieza Inicial...")

# Leer el CSV original
df = spark.read.csv(SOURCE_CSV_PATH, header=True, inferSchema=True)

# Columnas que queremos mantener inicialmente
columnas_mantener = [
    "ANIO", "MES", "CONGLOMERADO", "MUESTRA", "SELVIV", "HOGAR", "REGION", 
    "LLAVE_PANEL", "ESTRATO", "OCUP300", "INGTOT", 
    "C207", "C208", "C366", "C366_1", "C331", "C308_COD", "C309_COD"
]

# Filtrar y renombrar las columnas
df = (df.select(*columnas_mantener)
    .withColumnRenamed("ANIO", "anio")
    .withColumnRenamed("MES", "mes")
    .withColumnRenamed("CONGLOMERADO", "conglomerado")
    .withColumnRenamed("MUESTRA", "muestra")
    .withColumnRenamed("SELVIV", "selviv")
    .withColumnRenamed("HOGAR", "hogar")
    .withColumnRenamed("REGION", "region")
    .withColumnRenamed("LLAVE_PANEL", "llave_panel")
    .withColumnRenamed("ESTRATO", "estrato")
    .withColumnRenamed("OCUP300", "condicion_ocupacion")
    .withColumnRenamed("INGTOT", "ingreso_total")
    .withColumnRenamed("C207", "sexo")
    .withColumnRenamed("C208", "edad")
    .withColumnRenamed("C366", "nivel_educativo_cod")
    .withColumnRenamed("C366_1", "anios_educacion")
    .withColumnRenamed("C331", "horas_trabajadas")
    .withColumnRenamed("C308_COD", "ocupacion_cod")
    .withColumnRenamed("C309_COD", "sector_cod")
)

# ----------------------------------------------------------------------------------
# FASE 1.2: ELIMINAR COLUMNAS CON MUCHOS NULOS
# ----------------------------------------------------------------------------------
columnas_a_eliminar = ["horas_trabajadas", "ocupacion_cod", "sector_cod"]
df = df.drop(*columnas_a_eliminar)

print(f" Columnas eliminadas por alto % de nulos: {columnas_a_eliminar}")
print(" FASE 1 completada. Columnas restantes:")
print(df.columns)

# ----------------------------------------------------------------------------------
# GUARDAR DATASET FILTRADO
# ----------------------------------------------------------------------------------
OUTPUT_CSV_PATH = RUTAS_BASE + "dataset_filtrado.csv"
df.write.mode("overwrite").csv(OUTPUT_CSV_PATH, header=True)

print(f"\n Dataset filtrado guardado en: {OUTPUT_CSV_PATH}")
