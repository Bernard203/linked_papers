import numpy as np
import faiss

from sklearn.cluster import KMeans
from papers.models import Essay
from tqdm import tqdm


def load_feature_vectors():
    papers = Essay.objects.all()
    feature_vectors = []

    for paper in papers:
        # Restore feature vectors from binary field.
        vector = np.frombuffer(paper.feature_vector, dtype=np.float32)
        feature_vectors.append(vector)

    return np.array(feature_vectors)


def perform_clustering(feature_vectors, n_clusters=100):
    # Dimensions and distance indexes
    dim = feature_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(feature_vectors.astype(np.float32))

    # Clustering using KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(feature_vectors)

    # Returning clustering labels
    return kmeans.labels_


def cluster():
    feature_vectors = load_feature_vectors()
    cluster_labels = perform_clustering(feature_vectors)

    # Storing labels back to the database
    for i, paper in tqdm(enumerate(Essay.objects.all())):
        paper.cluster_label = cluster_labels[i]
        paper.save()
