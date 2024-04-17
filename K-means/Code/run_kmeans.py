import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib
from tqdm import tqdm

# Load patches
try:
    patches = np.load("patches.npy")
except FileNotFoundError:
    print("cd to folder Code, run extract_patches.py first.")
    exit()

K = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

# Perform K-means clustering
for n_clusters in tqdm(K, desc="Clustering"):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(patches)

    # Save the model
    joblib.dump(kmeans, f"../Result/Model/{n_clusters}-clusters-model.joblib")


# # Load the model
# model = joblib.load("../Result/Model/100-clusters-model.joblib")
