from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np

from Models.User_Model import User
from preprocessing import UserPreprocessor

# Constants for group size per cluster
MIN_MEMBERS = 4
MAX_MEMBERS = 10

def get_clusters(embeddings, feature_names, num_clusters=None):
    """
    Perform Gaussian Mixture Model clustering while respecting min/max member constraints.
    Applies higher weights to tag-related features.
    
    Args:
        embeddings: Feature matrix of shape (n_samples, n_features)
        feature_names: List of feature names corresponding to embedding columns
        num_clusters: Optional, specify exact number of clusters. If None, calculate optimal.
    
    Returns:
        Array of cluster assignments
    """
    if embeddings.size == 0 or len(embeddings.shape) <= 1:
        raise ValueError("Invalid embeddings shape for clustering")

    total_members = embeddings.shape[0]
    
    # Apply feature weights
    feature_weights = np.ones(len(feature_names))
    for i, feature in enumerate(feature_names):
        if feature.startswith('tag_'):
            feature_weights[i] = 3.0  # Give tags 3x importance
        elif feature.startswith('language_'):
            feature_weights[i] = 1.5  # Give languages 1.5x importance
    
    # Apply weights to features
    weighted_embeddings = embeddings * feature_weights
    
    # Calculate valid cluster range based on constraints
    max_possible_clusters = total_members // MIN_MEMBERS
    min_possible_clusters = max(1, (total_members + MAX_MEMBERS - 1) // MAX_MEMBERS)
    
    if num_clusters is None:
        # Choose optimal number of clusters within constraints
        num_clusters = min_possible_clusters
        
        # Try different numbers of clusters and pick the best one
        best_score = float('-inf')
        best_clusters = None
        
        # Scale features
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(weighted_embeddings)
        
        for n in range(min_possible_clusters, max_possible_clusters + 1):
            gmm = GaussianMixture(
                n_components=n,
                random_state=42,
                n_init=5,
                covariance_type='full'
            )
            gmm.fit(scaled_embeddings)
            
            # Get cluster assignments
            clusters = gmm.predict(scaled_embeddings)
            
            # Check if all clusters respect min/max constraints
            cluster_sizes = np.bincount(clusters)
            if np.all((cluster_sizes >= MIN_MEMBERS) & (cluster_sizes <= MAX_MEMBERS)):
                score = gmm.score(scaled_embeddings)
                if score > best_score:
                    best_score = score
                    best_clusters = clusters
        
        if best_clusters is not None:
            return best_clusters
            
    # If no valid clustering found or specific num_clusters given
    gmm = GaussianMixture(
        n_components=num_clusters,
        random_state=42,
        n_init=5,
        covariance_type='full'
    )
    
    # Scale features
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(weighted_embeddings)
    
    return gmm.fit_predict(scaled_embeddings)


if __name__ == "__main__":
    # Create sample users with different characteristics
    user1 = User(
        id=1,
        tags=["programming", "gaming"],
        country="USA",
        languages=["English", "Spanish"],
        preferred_group_size="small"
    )

    user2 = User(
        id=2,
        tags=["art", "music", "reading"],
        country="Canada",
        languages=["English", "French"],
        preferred_group_size="large"
    )

    user3 = User(
        id=3,
        tags=["sports", "gaming"],
        country="UK",
        languages=["English"],
        preferred_group_size="medium"
    )

    # Additional users
    user4 = User(
        id=4,
        tags=["programming", "art", "technology"],
        country="Germany",
        languages=["English", "German"],
        preferred_group_size="medium"
    )

    user5 = User(
        id=5,
        tags=["music", "gaming", "movies"],
        country="Japan",
        languages=["Japanese", "English"],
        preferred_group_size="small"
    )

    user6 = User(
        id=6,
        tags=["sports", "fitness", "health"],
        country="Australia",
        languages=["English"],
        preferred_group_size="large"
    )

    user7 = User(
        id=7,
        tags=["programming", "technology", "gaming"],
        country="USA",
        languages=["English"],
        preferred_group_size="medium"
    )

    user8 = User(
        id=8,
        tags=["art", "music", "movies"],
        country="France",
        languages=["French", "English"],
        preferred_group_size="small"
    )

    users = [user1, user2, user3, user4, user5, user6, user7, user8]

    print("=== Feature Extraction ===")
    preprocessor = UserPreprocessor()
    features = preprocessor.fit_transform(users)
    feature_names = preprocessor.get_feature_names()
    
    print(f"Total features extracted: {len(feature_names)}")
    print(f"Feature matrix shape: {features.shape}")
    print("\n=== User Features ===")
    for i, user in enumerate(users):
        print(f"\nUser {user.id}:")
        print(f"  Tags: {user.tags}")
        print(f"  Country: {user.country}")
        print(f"  Languages: {user.languages}")
        print(f"  Preferred group size: {user.preferred_group_size}")
        active_features = [feature_names[j] for j, val in enumerate(features[i]) if val == 1]
        print(f"  Active features: {active_features}")

    print("\n=== Clustering Results ===")
    # Perform clustering
    final_clusters = get_clusters(features, feature_names)
    
    # Organize users by cluster
    cluster_members = {}
    for i, cluster_id in enumerate(final_clusters):
        if cluster_id not in cluster_members:
            cluster_members[cluster_id] = []
        cluster_members[cluster_id].append(users[i])
    
    # Print cluster information
    for cluster_id, members in cluster_members.items():
        print(f"\nCluster {cluster_id}:")
        print(f"Size: {len(members)} members")
        print("Members:")
        for member in members:
            print(f"  - User {member.id}:")
            print(f"    Tags: {member.tags}")
            print(f"    Country: {member.country}")
            print(f"    Languages: {member.languages}")
            print(f"    Preferred group size: {member.preferred_group_size}")