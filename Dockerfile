FROM apache/spark:3.5.2-python3

USER root
WORKDIR /app


COPY ./app /app


RUN pip install --no-cache-dir -r requirements.txt || true

CMD ["spark-submit", "main.py"]
