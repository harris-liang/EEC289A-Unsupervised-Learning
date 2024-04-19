# Plot the centroids of the clusters
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

patch_size=5


K = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

# Perform K-means clustering
for n_clusters in tqdm(K, desc="Clustering"):
    # Load the model
    model = joblib.load("K-means/Result/Model/{}-clusters-model.joblib".format(n_clusters))

    fig, axs = plt.subplots(10, 10, figsize=(8, 8))  # Adjusted for a 10x10 grid of subplots
    axs = axs.ravel()  # Flatten the array of axes to simplify indexing

    for i in range(100):
        axs[i].imshow(model.cluster_centers_[i].reshape(patch_size, patch_size), cmap='gray')
        axs[i].axis('off')


    plt.savefig("K-means/Result/Centroids/{}-clusters-centroids.png".format(n_clusters), dpi=300)
    plt.close()