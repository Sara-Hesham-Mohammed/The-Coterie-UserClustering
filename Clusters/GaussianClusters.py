from sklearn.mixture import GaussianMixture


def get_clusters(embeddings, num_clusters=5):
    # Check if the reduced embeddings are empty
    if embeddings.size == 0:
        raise ValueError("The reduced embeddings are empty.")

    # Check if the reduced embeddings have more than 1 dimension
    if len(embeddings.shape) <= 1:
        raise ValueError("The reduced embeddings must have more than one dimension for clustering.")

    # Perform GMM clustering (soft clustering)
    n_clusters = num_clusters  # figure out way to make it dynamic according to how many users there are
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    clusters = gmm.fit_predict(embeddings)

    return clusters
