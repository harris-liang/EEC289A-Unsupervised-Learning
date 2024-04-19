import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import joblib


def visualize_reconstruction(n_clusters, patches, model):
    '''
    Visualize the original and reconstructed patches for a few random samples.
    
        @type   n_clusters: int
        @param  n_clusters: K

        @type   patches: ndarray
        @param  patches: patches

        @type   model: sklearn model
        @param  model: the fitted sklearn KMeans model
    '''
    # Predict the cluster for each patch
    labels = model.labels_

    # Get the cluster centers
    centroids = model.cluster_centers_

    # Pick random patches for display
    num_samples = 5  # Number of random samples to pick
    indices = np.random.choice(range(len(patches)), num_samples, replace=False)

    # Plotting the original and reconstructed patches
    fig, axs = plt.subplots(2, num_samples+1, figsize=(15, 3))  # 2 rows: originals and reconstructions

    # Set labels for the rows
    axs[0, 0].text(0.5, 0.5, 'Original', verticalalignment='center', horizontalalignment='center', transform=axs[0, 0].transAxes, fontsize = 15)
    axs[1, 0].text(0.5, 0.5, 'Reconstructed', verticalalignment='center', horizontalalignment='center', transform=axs[1, 0].transAxes, fontsize = 15)
    axs[0, 0].axis('off')
    axs[1, 0].axis('off')

    for i, idx in enumerate(indices):
        i += 1  # Adjust index for the extra label column

        # Original patches
        axs[0, i].imshow(patches[idx].reshape(5, 5), cmap='gray')
        axs[0, i].axis('off')
        axs[0, i].set_title('#{}'.format(idx), fontsize = 15)

        # Reconstructed patches
        reconstructed_patch = centroids[labels[idx]].reshape(5, 5)
        axs[1, i].imshow(reconstructed_patch, cmap='gray')
        axs[1, i].axis('off')
        axs[1, i].set_title('#{}'.format(labels[idx]), fontsize = 15)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"K-means/Result/Reconstruction/{n_clusters}-clusters-reconstruction.png", dpi=300)
    plt.close()


def mse_reconstruction(patches, model):
    '''
    Calculate the mean squared error between the original and reconstructed patches.

        @type   patches: ndarray
        @param  patches: patches

        @type   model: sklearn model
        @param  model: the fitted sklearn KMeans model
    '''
    # Predict the cluster for each patch
    labels = model.labels_

    # Get the cluster centers
    centroids = model.cluster_centers_

    # Calculate the mean squared error
    reconstruction = centroids[labels]
    mse = mean_squared_error(patches, reconstruction)

    return mse





if __name__ == "__main__":
    K = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    # Load patches
    try:
        patches = np.load("K-means/Code/patches.npy")
    except FileNotFoundError:
        print("cd to folder EEC289A, run extract_patches.py first.")
        exit()

    mse_K = []

    for n_clusters in K:
        # Load the model
        try:
            model = joblib.load(f"K-means/Result/Model/{n_clusters}-clusters-model.joblib")
        except FileNotFoundError:
            print(f"cd to folder EEC289A, run run_kmeans.py first.")
            exit()

        visualize_reconstruction(n_clusters, patches, model)

        mse = mse_reconstruction(patches, model)
        mse_K.append(mse)

    # Plot the mean squared error
    plt.plot(K, mse_K, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Mean Squared Error (MSE) of Patches Reconstruction')
    plt.title('MSE vs. Number of Clusters')
    plt.grid(True)
    plt.savefig("K-means/Result/Reconstruction/MSE_vs_K.png", dpi=300)