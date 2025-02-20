from sklearn.cluster import KMeans

def kmeans_clustering(data, k=5, random_state=42):
    """
    Applies K-Means clustering to the given dataset.

    Parameters:
    ----------
    data: array-like, shape (n_samples, n_features)
        The input data for clustering.
    
    k: int, default=5
        The number of clusters.
    
    random_state: int, default=42
        Random seed for reproducibility.

    Returns:
    -------
    labels: array, shape (n_samples,)
        Cluster labels for each data point.
    
    inertia: float
        Sum of squared distances of samples to their closest cluster center.
        
    """
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    labels = kmeans.fit_predict(data)
    return labels, kmeans.inertia_
