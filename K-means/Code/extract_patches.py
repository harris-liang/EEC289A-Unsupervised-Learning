from mnist_data_loader import MnistDataloader
from os.path  import join
import numpy as np
from tqdm import tqdm


# Extract 5x5 patches from the 28x28 images
def extract_patches(images, patch_size=5, threshold=0):
    '''
    Extract patches from images
    
        @type   images: ndarray
        @param  images: images
        
        @type   patch_size: int
        @param  patch_size: patch size
        
        @type   threshold: float
        @param  threshold: default 0 means non-blank patches from the training images
        
        @rtype:   ndarray
        @return:  patches
    '''
    patches = []
    num_pixels = patch_size * patch_size
    for image in tqdm(images, desc="Extracting patches"):
        # Slide over the image and extract patches
        for i in range(image.shape[0] - patch_size + 1):
            for j in range(image.shape[1] - patch_size + 1):
                patch = image[i:i + patch_size, j:j + patch_size]
                # Calculate the proportion of non-zero pixels
                if np.sum(patch) > 255 * num_pixels * threshold:  # Adjust the threshold as needed
                    patches.append(patch.flatten())
    return np.array(patches)


# Extract 5x5 patches from the 28x28 images
def extract_nonblank_patches_from_one_image(image, patch_size=5, threshold=0):
    '''
    Extract nonblank patches from one image
    
        @type   image: ndarray
        @param  image: image
        
        @type   patch_size: int
        @param  patch_size: patch size
        
        @type   threshold: float
        @param  threshold: default 0 means non-blank patches from the training images
        
        @rtype:   ndarray
        @return:  patches
    '''
    patches = []
    num_pixels = patch_size * patch_size
    # Slide over the image and extract patches
    for i in range(image.shape[0] - patch_size + 1):
        for j in range(image.shape[1] - patch_size + 1):
            patch = image[i:i + patch_size, j:j + patch_size]
            # Calculate the proportion of non-zero pixels
            if np.sum(patch) > 255 * num_pixels * threshold:  # Adjust the threshold as needed
                patches.append(patch.flatten())
    return np.array(patches)


def extract_all_patches_from_one_image(image, patch_size=5):
    '''
    Extract all patches (blank and nonblank) from one image
    
        @type   image: ndarray
        @param  image: image
        
        @type   patch_size: int
        @param  patch_size: patch size
        
        @rtype:   ndarray
        @return:  patches
    '''
    patches = []
    # Slide over the image and extract patches
    for i in range(image.shape[0] - patch_size + 1):
        for j in range(image.shape[1] - patch_size + 1):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch.flatten())
    return np.array(patches)



if __name__ == "__main__":
    # Set file paths based on added MNIST Datasets
    input_path = 'MNIST_ORG/'
    training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

    # Load MINST dataset
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


    # Extract non-blank patches from the training data
    patches = extract_patches(x_train)
    np.save("patches.npy", patches)