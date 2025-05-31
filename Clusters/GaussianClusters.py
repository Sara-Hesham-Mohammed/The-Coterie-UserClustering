from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def dimensionality_reduction(users_embeddings_df):
    # Check if the DataFrame is empty
    if users_embeddings_df.empty:
        raise ValueError("The embeddings DataFrame is empty.")

    # Check if the DataFrame has more than 1 column
    if users_embeddings_df.shape[1] <= 1:
        raise ValueError("The embeddings Dataframe must have more than one column for dimensionality reduction.")

    # Dimensionality Reduction (PCA) for better clustering
    pca = PCA(n_components=10)  # Reduce dimensions to 10
    reduced_embeddings = pca.fit_transform(users_embeddings_df)

    return reduced_embeddings

def get_clusters(reduced_embeddings, num_clusters=5):
    # Check if the reduced embeddings are empty
    if reduced_embeddings.size == 0:
        raise ValueError("The reduced embeddings are empty.")

    # Check if the reduced embeddings have more than 1 dimension
    if len(reduced_embeddings.shape) <= 1:
        raise ValueError("The reduced embeddings must have more than one dimension for clustering.")

    # Perform GMM clustering (soft clustering)
    n_clusters = num_clusters  # figure out way to make it dynamic according to how many users there are
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    clusters = gmm.fit_predict(reduced_embeddings)

    return clusters
