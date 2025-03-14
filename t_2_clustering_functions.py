import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def cluster(data, n_clusters=2):
    """
    Perform clustering on the given data_globus excluding the 'arrival_timestamp' column.

    Parameters:
        data (dict): The dataset as a dictionary.
        n_clusters (int): The number of clusters for K-Means.

    Returns:
        pd.DataFrame: The DataFrame with cluster labels.
    """
    # Step 1: Load the data_globus into a DataFrame
    df = pd.DataFrame(data)

    # Save the arrival timestamp
    arrival = df['arrival_timestamp']

    # Step 2: Select features for clustering (exclude 'arrival_timestamp')
    features = df.drop(columns=['arrival_timestamp'])
    column_names = features.columns.tolist()

    # Step 3: Normalize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Step 4: Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    # Step 5: Add the cluster labels to the DataFrame
    function_data = pd.DataFrame(scaled_features, columns=column_names)
    function_data['cluster'] = clusters

    # Step 6: Visualize the clusters (optional, for 2D data_globus)
    plt.scatter(features['argument_size'], features['loc'], c=clusters, cmap='viridis', s=50)
    plt.title('Clustering Visualization')
    plt.xlabel('Argument Size')
    plt.ylabel('LOC')
    plt.savefig('clustering_visualization.png')

    # Step 7: Recreate the dataset per 'function class'
    function_data = pd.concat([function_data, arrival], axis=1)

    for i in range(n_clusters):
        filtered_df = function_data[function_data['cluster'] == i]
        filtered_df = filtered_df.drop("cluster", axis=1)
        filtered_df.to_csv(f'data_globus/traces/endpoint1/functions/function{i}.csv', index=False)

if __name__ == "__main__":
    data = pd.read_csv("data_globus/traces/endpoint1/e1.csv")
    cluster(data, n_clusters=2)
