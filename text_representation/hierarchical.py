import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score

def plot_dendrogram(data, method='ward'):
    """
    Module for plotting a hierarchical clustering dendrogram.

    Parameters:
    -----------
    data : array-like
        The input data for hierarchical clustering, typically a feature matrix.

    method : str, optional (default='ward')
        The linkage method to use. Options include 'single', 'complete', 'average', and 'ward'.

    
    Returns:
    --------
    None
        Displays a plot showing a dendrogram.
    """

    linkage_matrix = linkage(data, method=method) 
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix)  
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.show()

def optimal_clusters(data, method='ward', cluster_method='both', max_k=10):
    """
    Finds the optimal number of clusters using the elbow method and/or silhouette score.

    Parameters:
    -----------
    data : array-like
        The input data for hierarchical clustering.

    method : str, optional (default='ward')
        The linkage method to use. Options include 'single', 'complete', 'average', and 'ward'.

    cluster_method : str (default='elbow')
        The method used to calculate the optimal number of clusters. 
        Options: 'elbow' (for Elbow Method), 'silhouette' (Silhouette Score), 'both' (returns both results).

    max_k : int (default=10)
        The maximum number of clusters to test for silhouette score.

    Returns:
    --------
    int or tuple
        The optimal number of clusters. If `cluster_method='both'`, returns a tuple (elbow_k, silhouette_k).
    """
    linkage_matrix = linkage(data, method=method)

    # Elbow Method
    distances = linkage_matrix[:, 2]  # Merge distances
    diffs = np.diff(distances)  # First derivative
    elbow_index = np.argmax(diffs) + 1  # +1 because `diff()` reduces the array length

    # Silhouette Score Method
    best_k = 2
    best_score = -1

    for k in range(2, min(max_k, len(data)) + 1):  # Avoid k > number of samples
        labels = fcluster(linkage_matrix, k, criterion='maxclust')
        score = silhouette_score(data, labels)
        if score > best_score:
            best_k = k
            best_score = score

    if cluster_method == 'elbow':
        return elbow_index
    elif cluster_method == 'silhouette':
        return best_k
    elif cluster_method == 'both':
        return elbow_index, best_k

if __name__ == "__main__":
    sample_data = np.random.rand(10, 4)  # Example random data
    plot_dendrogram(sample_data)
    print(f"Optimal clusters (Elbow Method): {optimal_clusters(sample_data, cluster_method='elbow')}")
    print(f"Optimal clusters (Silhouette Score): {optimal_clusters(sample_data, cluster_method='silhouette')}")
    print(f"Optimal clusters (Both Methods): {optimal_clusters(sample_data, cluster_method='both')}")