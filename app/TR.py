from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, desc

spark = SparkSession.builder.appName("ETL: Transformación e Inserción").getOrCreate()

# --- FUNCIONES DE IMPUTACIÓN ---
def imputar_estrato(df):
    """Imputa los nulos de estrato con la mediana global"""
    mediana = df.approxQuantile("estrato", [0.5], 0.0)[0]
    df = df.withColumn("estrato", when(col("estrato").isNull(), lit(mediana)).otherwise(col("estrato")))
    return df

def imputar_condicion_ocupacion(df):
    """Imputa los nulos de condicion_ocupacion con la moda global"""
    moda = df.groupBy("condicion_ocupacion").count().orderBy(desc("count")).first()[0]
    df = df.withColumn("condicion_ocupacion", when(col("condicion_ocupacion").isNull(), lit(moda)).otherwise(col("condicion_ocupacion")))
    return df

def imputar_ingreso_total(df):
    """Imputa ingreso_total solo para ocupados con edad >= 13"""
    grupos = df.select("sexo", "estrato").distinct().collect()
    for g in grupos:
        sexo_val = g["sexo"]
        estrato_val = g["estrato"]
        # Calcular mediana solo para filas con ingreso no nulo, edad >= 13 y ocupados
        mediana_grupo = df.filter(
            (col("sexo") == sexo_val) & 
            (col("estrato") == estrato_val) & 
            (col("ingreso_total").isNotNull()) &
            (col("edad") >= 13) &
            (col("condicion_ocupacion") == 1)
        ).approxQuantile("ingreso_total", [0.5], 0.0)[0]
        
        df = df.withColumn(
            "ingreso_total",
            when(
                (col("ingreso_total").isNull()) &
                (col("sexo") == sexo_val) &
                (col("estrato") == estrato_val) &
                (col("edad") >= 13) &
                (col("condicion_ocupacion") == 1),
                lit(mediana_grupo)
            ).otherwise(col("ingreso_total"))
        )
    
    # Para desocupados o edades < 13, forzar ingreso_total a NULL
    df = df.withColumn(
        "ingreso_total",
        when((col("edad") < 13) | (col("condicion_ocupacion") != 1), None).otherwise(col("ingreso_total"))
    )
    
    return df

def imputar_predictores(df):
    """Imputa nivel_educativo_cod y anios_educacion con la moda global"""
    columnas = ["nivel_educativo_cod", "anios_educacion"]
    for c in columnas:
        moda_global = df.groupBy(c).count().orderBy(desc("count")).first()[0]
        df = df.withColumn(c, when(col(c).isNull(), lit(moda_global)).otherwise(col(c)))
    return df

def map_ocupacion(df):
    """Convierte los valores de condicion_ocupacion a solo Ocupado / Desocupado"""
    df = df.withColumn(
        "condicion_ocupacion",
        when(col("condicion_ocupacion") == 1, "Ocupado").otherwise("Desocupado")
    )
    return df

def fase2_imputacion(df):
    """Aplica todas las imputaciones a las 5 columnas y mapea ocupacion"""
    df = imputar_estrato(df)
    df = imputar_condicion_ocupacion(df)
    df = imputar_ingreso_total(df)
    df = imputar_predictores(df)
    df = map_ocupacion(df)
    return df

# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    SOURCE_CSV_PATH = "/opt/spark-data/data_warehouse/dataset_filtrado.csv/dataset_filtrado.csv"
    OUTPUT_CSV_PATH = "/opt/spark-data/data_warehouse/dataset_limpio.csv"
    
    # Leer dataset filtrado
    df = spark.read.csv(SOURCE_CSV_PATH, header=True, inferSchema=True)
    
    print("\nAntes de la imputación:")
    df.select("estrato", "condicion_ocupacion", "ingreso_total", "nivel_educativo_cod", "anios_educacion").show(5)
    
    # Aplicar imputación y mapeo de ocupacion
    df = fase2_imputacion(df)
    
    print("\nDespués de la imputación y mapeo Ocupado/Desocupado:")
    df.select("estrato", "condicion_ocupacion", "ingreso_total", "nivel_educativo_cod", "anios_educacion").show(5)
    
    # Guardar dataset limpio
    df.write.mode("overwrite").csv(OUTPUT_CSV_PATH, header=True)
    print(f"\n✅ Dataset limpio guardado en: {OUTPUT_CSV_PATH}")
