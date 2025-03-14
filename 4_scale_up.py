from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.ml.feature import StandardScalerModel

def reverse_scaling(input_path, scaler_model_path):
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("Reverse Scaling") \
        .getOrCreate()

    # Load the scaled data_globus
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Load the saved StandardScalerModel
    scaler_model = StandardScalerModel.load(scaler_model_path)

    # Retrieve the standard deviation and mean vectors from the scaler model
    std_dev = scaler_model.std.toArray()
    mean = scaler_model.mean.toArray()  # Use only if mean centering was enabled

    # Reverse scaling for each column:
    features_to_inverse = ["actuals", "predictions"]
    # select column 'n_invocations' corresponding index in the original scaled data_globus
    i = 0
    # actuals
    print(f"Reversing scaling for column: {features_to_inverse[0]}")
    df = df.withColumn(features_to_inverse[0], col(features_to_inverse[0]) * std_dev[i] + mean[i]) #Reversing
    # predictions
    print(f"Reversing scaling for column: {features_to_inverse[1]}")
    df = df.withColumn(features_to_inverse[1], col(features_to_inverse[1]) * std_dev[i] + mean[i]) #Reversing

    # Save the reversed data_globus to a CSV
    df.coalesce(1).write.csv("data_globus/results", header=True, mode="overwrite")

    # Stop the Spark session
    spark.stop()


if __name__ == "__main__":
    scaled_csv_path = "data_globus/results.csv"
    scaler_model_path = "helpers/scaler/"

    # Step 2: Reverse the scaling
    reverse_scaling(scaled_csv_path, scaler_model_path)