import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib
from os.path  import join

from mnist_data_loader import MnistDataloader
from extract_patches import extract_all_patches_from_one_image, extract_nonblank_patches_from_one_image



def calculate_positions(image_shape, patch_size):
    '''
    Calculate the positions of all patches in an image
    
        @type   image_shape: tuple
        @param  image_shape: image shape
        
        @type   patch_size: int
        @param  patch_size: patch size
        
        @rtype:   list
        @return:  positions
    '''
    positions = []
    for i in range(image_shape[0] - patch_size + 1):
        for j in range(image_shape[1] - patch_size + 1):
            positions.append((i, j))
    return positions



def reconstruct_digit(digit_image, model, patch_size = 5):
    '''
    Reconstruct a digit image using a KMeans model

        @type   digit_image: ndarray
        @param  digit_image: digit image

        @type   model: sklearn model
        @param  model: KMeans model

        @type   patch_size: int
        @param  patch_size: patch size

        @rtype:   ndarray
        @return:  reconstructed image
    '''
    # Extract all patches and calculate positions
    all_patches = extract_all_patches_from_one_image(digit_image)
    positions = calculate_positions(digit_image.shape, patch_size)

    # Extract non-blank patches
    nonblank_patches = extract_nonblank_patches_from_one_image(digit_image)
    
    # Map nonblank patches to their positions
    nonblank_indices = [i for i, patch in enumerate(all_patches) if np.sum(patch) > 0]

    # Predict clusters for non-blank patches
    labels = model.predict(nonblank_patches)

    # Initialize the reconstructed image with zeros
    reconstructed_image = np.zeros_like(digit_image, dtype=float)
    count_matrix = np.zeros_like(digit_image, dtype=float)

    # Add centroids to the corresponding positions
    centroids = model.cluster_centers_
    for label, idx in zip(labels, nonblank_indices):
        i, j = positions[idx]
        reconstructed_image[i:i+patch_size, j:j+patch_size] += centroids[label].reshape(patch_size, patch_size)
        count_matrix[i:i+patch_size, j:j+patch_size] += 1

    # Avoid division by zero
    count_matrix[count_matrix == 0] = 1
    reconstructed_image /= count_matrix

    return reconstructed_image

    



if __name__ == "__main__":
    # Set file paths based on added MNIST Datasets
    input_path = 'K-means/Code/MNIST_ORG/'
    training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')


    # Load MINST dataset
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


    # Example of reconstructing a digit
    digit_idx = np.random.randint(0, len(x_test))
    digit_image = x_test[digit_idx]



    K = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]


    # Prepare the figure for subplots
    fig, axs = plt.subplots(2, 10, figsize=(18, 4))  # 1 row, columns for each K plus one for the original

    # Display the original digit in the first column
    axs[0, 0].imshow(digit_image, cmap='gray')
    axs[0, 0].set_title('Original Number {}'.format(y_train[digit_idx]))
    axs[0, 0].axis('off')



    for i_fig, n_clusters in enumerate(K):
        # Load the model
        try:
            model = joblib.load(f"K-means/Result/Model/{n_clusters}-clusters-model.joblib")
        except FileNotFoundError:
            print(f"cd to folder EEC289A, run run_kmeans.py first.")
            exit()

        # Reconstruct the digit
        reconstructed_image = reconstruct_digit(digit_image, model)

        # Display reconstructed digit for this K
        if i_fig < 9:
            axs[0, i_fig + 1].imshow(reconstructed_image, cmap='gray')
            axs[0, i_fig + 1].set_title(f'K={n_clusters}')
            axs[0, i_fig + 1].axis('off')
        else:
            axs[1, i_fig - 9].imshow(reconstructed_image, cmap='gray')
            axs[1, i_fig - 9].set_title(f'K={n_clusters}')
            axs[1, i_fig - 9].axis('off')

    # plt.tight_layout()
    plt.savefig(f"K-means/Result/Digits/reconstruct-test-digit.png", dpi=300)
