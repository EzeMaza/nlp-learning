from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def silhouette_method(data, max_k=10):
    """
    Determines the optimal number of clusters for K-Means clustering using the Silhouette Score.

    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        The dataset to cluster.

    max_k : int, optional (default=10)
        The maximum number of clusters to consider. Must be at least 2, since the silhouette score is undefined for k=1.

    Returns:
    --------
    None
        Displays a plot showing the silhouette score for different cluster sizes.

    Notes:
    ------
    The silhouette score measures how well samples are clustered. Higher values indicate better-defined clusters.
    The optimal number of clusters is often where the silhouette score is maximized.
    """

    scores = []
    ks = range(2, max_k + 1)  # Silhouette Score is not defined for k=1

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append(score)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, scores, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.show()