import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def distance_plot(data, k=4, metric='cosine'):
  """
  Plots the k-distance graph to help determine the optimal epsilon value 
  for DBSCAN clustering using the elbow method. 

  Parameters:
  -----------
  data : array-like, shape (n_samples, n_features)
    The dataset to analyze. 

  k : int, optional (default=4)
    The number of nearest neighbours. A rule of thumb is to use k = 2 * MinPts

  metric: str, optional (default='cosine')
    The choosen metric to calculate distances
        

  Returns:
  --------
  None
    Displays a plot showing the sorted k-th nearest neighbor distances.

  Notes:
  ------
  - This plot is commonly used to find the appropriate `eps` parameter for 
    DBSCAN clustering. The point of maximum curvature (elbow) indicates a 
    reasonable choice for `eps`.
  - Choosing an appropriate `k` value depends on the dataset; for small 
    datasets, `k=4` is often used, while larger datasets might require a 
    higher value.
  """
    
  neigh = NearestNeighbors(n_neighbors=k,  metric=metric)
  neigh.fit(data)
  distances, _ = neigh.kneighbors(data)

    
  k_distances = np.sort(distances[:, k-1])        # Sort distances

  plt.figure(figsize=(8, 5))
  plt.plot(k_distances, marker='o')
  plt.xlabel("Points sorted by distance")
  plt.ylabel(f"{k}-th Nearest Neighbor Distance")
  plt.title(f"K-Distance Plot for k={k}")
  plt.grid(True)
  plt.show()
