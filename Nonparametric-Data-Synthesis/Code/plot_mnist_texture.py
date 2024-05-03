import numpy as np
import matplotlib.pyplot as plt
from mnist_data_loader import MnistDataloader
from os.path  import join

def create_mnist_collage(x_train, N=10):
    '''
    Create a collage of MNIST images

        @type   x_train: ndarray
        @param  x_train: Training images

        @type   N: int
        @param  N: Number of images along each dimension (NxN)
    '''
    # Number of images along each dimension (NxN)
    indices = np.random.choice(len(x_train), N*N, replace=False)  # Randomly pick indices
    images = x_train[indices]  # Extract the corresponding images

    # # Create an empty array to hold the entire NxN collage
    # collage = np.zeros((28*N, 28*N))

    # # Fill the collage array with MNIST images
    # for i in range(N):
    #     for j in range(N):
    #         collage[i*28:(i+1)*28, j*28:(j+1)*28] = images[i*N+j]

    # Create a new array that stacks images in N rows, each containing N images
    # First reshape images to (N, N, 28, 28), then transpose to (N, 28, N, 28) and reshape to final (28*N, 28*N)
    collage = images.reshape(N, N, 28, 28).transpose(0, 2, 1, 3).reshape(28*N, 28*N)

    # Display the collage
    plt.figure(figsize=(28*N/100, 28*N/100), dpi=100, frameon=False)  # Set figure size without frame
    plt.imshow(collage, cmap='gray', aspect='auto')  # Show image in grayscale, adjust aspect ratio
    plt.axis('off')  # Hide axes
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Remove x-axis locator
    plt.gca().yaxis.set_major_locator(plt.NullLocator())  # Remove y-axis locator
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # Adjust subplot parameters
    plt.margins(0, 0)  # Set margins to zero
    plt.savefig('Nonparametric-Data-Synthesis/Code/Textures/mnist.png')  # Save the collage



if __name__ == '__main__':
    # Set file paths based on added MNIST Datasets
    input_path = 'K-means/Code/MNIST_ORG/'
    training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

    # Load MINST dataset
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # Create and display the collage
    create_mnist_collage(x_train, N=20)