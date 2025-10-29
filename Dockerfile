FROM apache/spark:3.5.2-python3

USER root
WORKDIR /app

# Copiar la app local dentro del contenedor
COPY ./app /app

# Instalar dependencias adicionales (si las hay)
RUN pip install --no-cache-dir -r requirements.txt || true

CMD ["spark-submit", "main.py"]
