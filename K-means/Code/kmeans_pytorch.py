import torch
from tqdm import tqdm

def kmeans_pytorch(X, n_clusters, n_iters=100, tolerance=1e-4):
    """
    Performs k-means clustering using PyTorch.

    Parameters:
        X (torch.Tensor): The input data, a tensor of shape (n_samples, n_features).
        n_clusters (int): The number of clusters to form.
        n_iters (int): Maximum number of iterations of the k-means algorithm.
        tolerance (float): Tolerance to declare convergence.

    Returns:
        centers (torch.Tensor): Cluster centers, a tensor of shape (n_clusters, n_features).
        labels (torch.Tensor): Index of the cluster each sample belongs to.
    """
    # Randomly choose cluster centers from the input data at the start.
    indices = torch.randperm(X.size(0))[:n_clusters]
    centers = X[indices]

    for _ in tqdm(range(n_iters), desc="K-means"):
        # Compute distances from data points to the centroids
        distances = torch.cdist(X, centers)
        # Assign clusters
        labels = torch.argmin(distances, dim=1)
        # Compute new centers
        new_centers = torch.stack([X[labels == i].mean(dim=0) for i in range(n_clusters)])

        # Check for convergence
        if torch.norm(centers - new_centers) < tolerance:
            break
        
        centers = new_centers
    
    return centers, labels


if __name__ == "__main__":
    # Example usage
    # Creating some data
    torch.manual_seed(0)
    data = torch.randn(100, 2)  # 100 data points, 2 dimensions

    # Clustering
    centers, labels = kmeans_pytorch(data, n_clusters=3)
    print("Cluster centers:\n", centers)
    print("Cluster labels:\n", labels)
