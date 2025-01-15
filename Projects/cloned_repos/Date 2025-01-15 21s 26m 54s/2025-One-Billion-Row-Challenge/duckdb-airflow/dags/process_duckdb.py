import duckdb
import time
from airflow.decorators import dag, task
from datetime import datetime

@dag(
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={"owner": "airflow"},
    tags=["duckdb"],
)
def duckdb_airflow_dag():

    @task
    def create_duckdb():
        conn = duckdb.connect(database=":memory:")
        conn.execute("PRAGMA threads=16;")  # Usa 16 núcleos
        conn.execute("PRAGMA memory_limit='12GB';")  # Usa mais memória

        duckdb.sql("""
            SELECT station,
                MIN(temperature) AS min_temperature,
                CAST(AVG(temperature) AS DECIMAL(3,1)) AS mean_temperature,
                MAX(temperature) AS max_temperature
            FROM read_parquet('/usr/local/airflow/include/medicoes_1000000000.parquet/*.parquet')
            GROUP BY station
            ORDER BY station
        """).show()

    create_duckdb()

duckdb_airflow_dag()
