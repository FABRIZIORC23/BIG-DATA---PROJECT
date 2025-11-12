from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, lit, log, when, sha2

# --- CONFIGURACIÓN Y SESIÓN ---
# Crear la sesión de Spark
spark = SparkSession.builder.appName("ETL: Extraccion, Transformacion y Carga (DW)").getOrCreate()
RUTAS_BASE = "/opt/spark-data/data_warehouse/"
SOURCE_CSV_PATH = "/opt/spark-data/dataset.csv"

# ----------------------------------------------------------------------------------
# FASE 1: EXTRACCIÓN Y LIMPIEZA INICIAL (E)
# ----------------------------------------------------------------------------------

print("\n🚀 FASE 1: Extracción y Limpieza Inicial...")
# Leer el CSV original
df = spark.read.csv(SOURCE_CSV_PATH, header=True, inferSchema=True)

# Columnas que queremos mantener (basado en la Tabla 1 del documento)
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
print("✅ FASE 1: Extracción y Limpieza Inicial completada.")

# ----------------------------------------------------------------------------------
# FASE 2: TRANSFORMACIÓN (T)
# ----------------------------------------------------------------------------------

print("⏳ FASE 2: Transformación y Enriquecimiento de Datos...")

# 2.1. Creación de Identificadores Únicos (Para las llaves primarias y foráneas)
df = df.withColumn("id_tiempo", concat(col("anio"), lit("-"), col("mes")))
df = df.withColumn("id_hogar", concat(col("conglomerado"), lit("-"), col("muestra"), lit("-"), col("selviv"), lit("-"), col("hogar")))
df = df.withColumn("id_persona", concat(col("id_hogar"), lit("-"), col("llave_panel")))

# 2.2. Filtrado y Limpieza de Variables Críticas (Edad, Sexo, Condición de Ocupación)
df_limpio = df.na.drop(subset=["edad", "sexo", "condicion_ocupacion"])

# 2.3. Generación de Variables Derivadas y Objetivos
df_limpio = df_limpio.withColumn("log_ingreso_total", log(col("ingreso_total") + 1))
df_limpio = df_limpio.withColumn("ingreso_por_hora", when(col("horas_trabajadas") > 0, col("ingreso_total") / (col("horas_trabajadas") * 4)).otherwise(0))
df_limpio = df_limpio.withColumn("experiencia_laboral_potencial", when((col("edad") - col("anios_educacion") - lit(6)) > 0, col("edad") - col("anios_educacion") - lit(6)).otherwise(0))

# Variable Objetivo: Clasificación (1=Ocupado, 0=Desocupado. Se excluyen los Inactivos)
df_limpio = df_limpio.withColumn("ocupado_binario", 
    when(col("condicion_ocupacion") == 1, 1) # Ocupado
    .when(col("condicion_ocupacion") == 3, 0) # Desocupado (asumiendo que 3 es desocupado en tu muestra)
    .otherwise(None) # Excluye Inactivo (4) y otros
)
df_limpio = df_limpio.na.drop(subset=["ocupado_binario"]) 

# 2.4. Agrupación y Manejo de Faltantes
# Agrupación del Nivel Educativo (Sin Educacion, Basica, Superior)
df_limpio = df_limpio.withColumn("nivel_agrupado",
    when((col("nivel_educativo_cod") <= 2) | (col("nivel_educativo_cod").isNull()), lit("Sin_Educacion"))
    .when((col("nivel_educativo_cod") >= 3) & (col("nivel_educativo_cod") <= 7), lit("Basica"))
    .otherwise(lit("Superior"))
)
# Imputación de horas de trabajo faltantes con 40 horas
df_limpio = df_limpio.na.fill(value=40, subset=["horas_trabajadas"])
# Imputación de Ingreso Total (0 si no está ocupado)
df_limpio = df_limpio.withColumn("ingreso_total_imputado", when(col("condicion_ocupacion") != 1, 0).otherwise(col("ingreso_total")))

# Creación de Claves Subrogadas para las dimensiones (Ubicacion y Empleo)
df_limpio = df_limpio.withColumn("id_ubicacion_hash", sha2(concat(col("region"), col("estrato")), 256))
df_limpio = df_limpio.withColumn("id_empleo_hash", sha2(concat(col("ocupacion_cod"), col("sector_cod")), 256))

df_final = df_limpio.cache()
print("✅ FASE 2: Transformación completa. DataFrame enriquecido.")

# ----------------------------------------------------------------------------------
# FASE 3: CARGA (L) - Creación del Modelo Estrella
# ----------------------------------------------------------------------------------

print("📦 FASE 3: Carga y Creación del Modelo Estrella...")

# Función auxiliar para guardar las dimensiones
def save_dimension(df, cols, dim_name):
    df.select(*cols).distinct().write.mode("overwrite").parquet(RUTAS_BASE + dim_name)
    print(f"   -> {dim_name} cargada.")

# 3.1. Creación y Carga de las Tablas de Dimensión
save_dimension(df_final, ["id_tiempo", "anio", "mes"], "dim_tiempo")
save_dimension(df_final, ["id_persona", "sexo", "edad"], "dim_persona")
save_dimension(df_final, ["id_persona", "nivel_agrupado", "anios_educacion"], "dim_educacion")
save_dimension(df_final, ["id_ubicacion_hash", "region", "estrato"], "dim_ubicacion")
save_dimension(df_final, ["id_hogar", "conglomerado", "muestra", "selviv", "hogar"], "dim_hogar")
save_dimension(df_final, ["id_empleo_hash", "ocupacion_cod", "sector_cod"], "dim_empleo")

# 3.2. Creación y Carga de la Tabla de Hechos
fact_cols = [
    "id_tiempo", "id_persona", "id_hogar", "id_ubicacion_hash", "id_empleo_hash",
    "condicion_ocupacion", "ocupado_binario", "ingreso_total_imputado",
    "log_ingreso_total", "ingreso_por_hora", "experiencia_laboral_potencial"
]
fact_table = df_final.select(*fact_cols)

# Guardar la tabla de hechos, particionada por id_tiempo
fact_table.write.partitionBy("id_tiempo").mode("overwrite").parquet(RUTAS_BASE + "fact_empleo_ingresos_final")
print("✅ FASE 3: Modelo Estrella cargado. Tabla de Hechos finalizada.")

df_final.unpersist()
spark.stop()