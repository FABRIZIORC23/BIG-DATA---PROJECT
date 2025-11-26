# CARG.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sha2, concat, lit

# --- CONFIGURACIÓN Y SESIÓN ---
spark = SparkSession.builder.appName("ETL: Carga DW").getOrCreate()
RUTAS_BASE = "/opt/spark-data/data_warehouse/"
SOURCE_CSV_PATH = RUTAS_BASE + "dataset_limpio.csv/dataset_limpio.csv"

print(" FASE 3: Carga y Creación del Modelo Estrella...")

# Leer dataset limpio
df = spark.read.csv(SOURCE_CSV_PATH, header=True, inferSchema=True)

# ----------------------------------------------------------------------------------
# 3.0 Generar IDs para el modelo estrella
# ----------------------------------------------------------------------------------

# id_tiempo
df = df.withColumn("id_tiempo", sha2(concat(col("anio"), lit("-"), col("mes")), 256))

# id_persona
df = df.withColumn("id_persona", sha2(concat(col("llave_panel"), lit("-"), col("hogar")), 256))

# id_ubicacion_hash
df = df.withColumn("id_ubicacion_hash", sha2(concat(col("region"), lit("-"), col("estrato")), 256))

# id_hogar
df = df.withColumn("id_hogar", sha2(concat(col("conglomerado"), lit("-"), col("muestra"), lit("-"), col("hogar")), 256))

# id_empleo_hash (sin ocupacion_cod ni sector_cod)
df = df.withColumn("id_empleo_hash", sha2(concat(col("condicion_ocupacion"), lit("-"), col("estrato")), 256))

# ----------------------------------------------------------------------------------
# 3.1 Función auxiliar para guardar dimensiones
# ----------------------------------------------------------------------------------
def save_dimension(df, cols, dim_name):
    df.select(*cols).distinct().write.mode("overwrite").parquet(RUTAS_BASE + dim_name)
    print(f"   -> {dim_name} cargada.")

# 3.2 Creación y carga de las tablas de dimensión
save_dimension(df, ["id_tiempo", "anio", "mes"], "dim_tiempo")
save_dimension(df, ["id_persona", "sexo", "edad"], "dim_persona")
save_dimension(df, ["id_persona", "nivel_educativo_cod", "anios_educacion"], "dim_educacion")
save_dimension(df, ["id_ubicacion_hash", "region", "estrato"], "dim_ubicacion")
save_dimension(df, ["id_hogar", "conglomerado", "muestra", "selviv", "hogar"], "dim_hogar")
save_dimension(df, ["id_empleo_hash", "condicion_ocupacion"], "dim_empleo")

# ----------------------------------------------------------------------------------
# 3.3 Creación de la tabla de hechos
# ----------------------------------------------------------------------------------
fact_cols = [
    "id_tiempo",
    "id_persona",
    "id_hogar",
    "id_ubicacion_hash",
    "id_empleo_hash",
    "condicion_ocupacion",
    "ingreso_total",
    "estrato",
    "nivel_educativo_cod",
    "anios_educacion"
]

fact_table = df.select(*fact_cols)

# Guardar la tabla de hechos particionada por tiempo
fact_table.write.partitionBy("id_tiempo").mode("overwrite").parquet(RUTAS_BASE + "fact_empleo_ingresos_final")
print(" FASE 3: Modelo Estrella cargado. Tabla de Hechos finalizada.")

df.unpersist()
spark.stop()
