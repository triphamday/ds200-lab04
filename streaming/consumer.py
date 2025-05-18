from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType
from streaming.config import HOST, PORT, PARQUET_PATH, CHECKPOINT_PATH

def run_stream_consumer():
    spark = SparkSession.builder \
        .appName("PokemonStreamConsumer") \
        .getOrCreate()

    schema = StructType([
        StructField("path", StringType(), True),
        StructField("label", StringType(), True)
    ])

    raw = spark.readStream \
        .format("socket") \
        .option("host", HOST) \
        .option("port", PORT) \
        .load()

    df = raw.select(from_json(col("value"), schema).alias("data")) \
            .select("data.path", "data.label")

    query = df.writeStream \
        .format("parquet") \
        .option("path", PARQUET_PATH) \
        .option("checkpointLocation", CHECKPOINT_PATH) \
        .trigger(processingTime='30 seconds') \
        .outputMode("append") \
        .start()

    print("Spark consumer started, writing parquet...")
    query.awaitTermination()

if __name__ == '__main__':
    run_stream_consumer()
