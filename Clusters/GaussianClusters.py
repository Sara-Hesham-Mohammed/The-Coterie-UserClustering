from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict


def get_clusters(users_data, num_clusters=5):
    """
    Perform Gaussian Mixture Model (GMM) clustering on user embeddings, considering both
    interest and language/country features separately and then combining their probabilities.
    
    Args:
        users_data: list of user data dictionaries containing interest_embedding and location_lang_embedding
        num_clusters: number of clusters to form (default is 5).
    Returns:
        dict: A dictionary where keys are cluster IDs (0 to num_clusters-1) and values are lists of users in that cluster.
    """
    # Extract features from users_data
    interest_embeddings = np.array([user['interest_embedding'] for user in users_data])
    lang_country_embeddings = np.array([user['location_lang_embedding'] for user in users_data])
    
    # Validate the embeddings
    if interest_embeddings.size == 0 or lang_country_embeddings.size == 0:
        raise ValueError("The embeddings cannot be empty.")
    
    if len(interest_embeddings.shape) <= 1 or len(lang_country_embeddings.shape) <= 1:
        raise ValueError("The embeddings must have more than one dimension for clustering.")
    
    # Standardize both feature sets
    scaler_interests = StandardScaler()
    scaler_lang = StandardScaler()
    
    interest_embeddings_scaled = scaler_interests.fit_transform(interest_embeddings)
    lang_country_embeddings_scaled = scaler_lang.fit_transform(lang_country_embeddings)
    
    # Create and fit GMMs for both feature sets
    gmm_interests = GaussianMixture(n_components=num_clusters, random_state=42)
    gmm_lang = GaussianMixture(n_components=num_clusters, random_state=42)
    
    # Fit models and get probability distributions for each feature set
    gmm_interests.fit(interest_embeddings_scaled)
    gmm_lang.fit(lang_country_embeddings_scaled)
    
    interest_probs = gmm_interests.predict_proba(interest_embeddings_scaled)
    lang_probs = gmm_lang.predict_proba(lang_country_embeddings_scaled)
    
    # Combine probabilities with equal weights (0.5 each)
    combined_probs = 0.5 * interest_probs + 0.5 * lang_probs
    
    # Get cluster assignments for each user
    cluster_labels = np.argmax(combined_probs, axis=1)

    print(f"Cluster labels: {cluster_labels}, type: {type(cluster_labels)}")
    
    # Create a dictionary to store users by cluster
    clusters_dict = defaultdict(list)
    
    # Group users by their cluster
    # Group users by their cluster
    for user_idx, cluster_label in enumerate(cluster_labels):
        user_id = users_data[user_idx]['id']
        clusters_dict[int(cluster_label)].append(user_id)

    # Convert defaultdict to regular dict
    return dict(clusters_dict)
