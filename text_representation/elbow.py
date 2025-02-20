import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def elbow_method(data, max_k=10):
    """
    Determines the optimal number of clusters for K-Means clustering using the Elbow Method.
    
    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        The dataset to cluster.
    
    max_k : int, optional (default=10)
        The maximum number of clusters to consider.

    Returns:
    --------
    None
        Displays a plot showing the inertia for different cluster sizes.
    
    Notes:
    ------
    The "elbow point" in the plot is where the inertia starts decreasing at a slower rate,
    indicating an optimal number of clusters.
    """
    inertias = []
    ks = range(1, max_k + 1)

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()