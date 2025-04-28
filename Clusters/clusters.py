from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

users = [
    {"id": 1, "age": 25, "tags": ["sports", "tech", "music", "gaming", "travel"]},
    {"id": 2, "age": 40, "tags": ["cooking", "music", "tech", "finance", "reading"]},
    {"id": 3, "age": 33, "tags": ["travel", "photography", "tech", "fitness", "music"]},
    {"id": 4, "age": 22, "tags": ["gaming", "tech", "anime", "music", "sports"]},
    {"id": 5, "age": 55, "tags": ["gardening", "cooking", "reading", "history", "travel"]},
    {"id": 6, "age": 29, "tags": ["music", "travel", "sports", "fitness", "food"]},
    {"id": 7, "age": 61, "tags": ["history", "reading", "gardening", "cooking", "finance"]},
    {"id": 8, "age": 45, "tags": ["finance", "tech", "sports", "news", "reading"]},
    {"id": 9, "age": 37, "tags": ["fitness", "cooking", "music", "photography", "health"]},
    {"id": 10, "age": 19, "tags": ["anime", "gaming", "sports", "tech", "memes"]},
    {"id": 11, "age": 48, "tags": ["travel", "history", "finance", "tech", "reading"]},
    {"id": 12, "age": 31, "tags": ["tech", "music", "gaming", "food", "anime"]}
]


# Helper function to find common tags between a user and a cluster
def get_common_tags(user, cluster):
    user_tags = set(user["tags"])
    if not cluster:
        return user_tags

    cluster_common_tags = set.intersection(*[set(u["tags"]) for u in cluster])
    return user_tags.intersection(cluster_common_tags)

def age_classification(user):
    # Classify age into groups
    if 17 <= user["age"] < 20:
        return "teen"
    elif 20 <= user["age"] < 30:
        return "young_adult"
    elif 30 <= user["age"] < 50:
        return "adult"
    elif 50 <= user["age"] < 65:
        return "senior"
    else:
        "child: participation not allowed"

# Modified is_valid to ensure at least one common tag
def is_valid(user, cluster):
    if not cluster:  # If cluster is empty, it's valid
        return True

    # Check age constraint
    age = age_classification(user)
    jaccard_score()

    # Check tag similarity constraints
    for other in cluster:
        sim = jaccard(user["tags"], other["tags"])
        if sim > 0.7 or sim < 0.2:  # Constraint 2: tag similarity constraints
            return False

    # Check if there's at least one common tag with the entire cluster
    common_tags = get_common_tags(user, cluster)
    if not common_tags:
        return False

    return True


def jaccard(set1, set2):
    # Fix: Convert lists to sets and use proper set operations
    set1 = set(set1)
    set2 = set(set2)
    return len(set1.intersection(set2)) / len(set1.union(set2))


def get_clusters(users, initial_clusters=3, min_cluster_size=4):
    # Convert tags to vectors for initial clustering
    mlb = MultiLabelBinarizer()
    tag_matrix = mlb.fit_transform([user["tags"] for user in users])

    # Add age as a feature (normalized)
    ages = np.array([user["age"] for user in users]).reshape(-1, 1)
    ages_normalized = (ages - ages.mean()) / ages.std()

    # Combine features
    features = np.hstack([tag_matrix, ages_normalized])

    # Get initial cluster suggestions using KMeans
    kmeans = KMeans(n_clusters=initial_clusters)
    suggested_labels = kmeans.fit_predict(features)

    # Create initial suggested clusters
    suggested_clusters = [[] for _ in range(initial_clusters)]
    for user, label in zip(users, suggested_labels):
        suggested_clusters[label].append(user)

    # Final clusters that respect constraints
    final_clusters = []

    # Process each suggested cluster
    for suggested_cluster in suggested_clusters:
        valid_sub_cluster = []

        # Try to keep users together from the suggested cluster if they're valid
        for user in suggested_cluster:
            # Check if adding this user would maintain at least one common tag
            would_have_common_tag = bool(get_common_tags(user, valid_sub_cluster))

            if is_valid(user, valid_sub_cluster) and (not valid_sub_cluster or would_have_common_tag):
                valid_sub_cluster.append(user)
            else:
                # Try adding to existing final clusters
                added = False
                for cluster in final_clusters:
                    would_have_common_tag = bool(get_common_tags(user, cluster))
                    if is_valid(user, cluster) and would_have_common_tag:
                        cluster.append(user)
                        added = True
                        break

                # Create new cluster if needed
                if not added:
                    final_clusters.append([user])

        # Add the valid sub-cluster if it's not empty
        if valid_sub_cluster:
            final_clusters.append(valid_sub_cluster)

    # Process any remaining users who didn't fit well
    processed_users = set(u["id"] for cluster in final_clusters for u in cluster)
    remaining_users = [u for u in users if u["id"] not in processed_users]

    if remaining_users:
        # Create new clusters for remaining users that maintain common tags
        current_cluster = []
        for user in remaining_users:
            if not current_cluster or get_common_tags(user, current_cluster):
                if is_valid(user, current_cluster):
                    current_cluster.append(user)
                    continue

            # If we can't add to current_cluster, try existing clusters
            added = False
            for cluster in final_clusters:
                would_have_common_tag = bool(get_common_tags(user, cluster))
                if is_valid(user, cluster) and would_have_common_tag:
                    cluster.append(user)
                    added = True
                    break

            # If still not added, start a new cluster if current one has users
            if not added:
                if current_cluster:
                    final_clusters.append(current_cluster)
                    current_cluster = [user]
                else:
                    current_cluster = [user]

        # Add the last cluster if it has users
        if current_cluster:
            final_clusters.append(current_cluster)

    # Special merging function that preserves common tags constraint
    def can_merge_clusters(cluster1, cluster2):
        # Check if merged cluster would have at least one common tag
        merged_common_tags = set.intersection(*[set(u["tags"]) for u in cluster1 + cluster2])
        if not merged_common_tags:
            return False

        # Check if all users would still be valid with everyone else
        for user in cluster1:
            for other in cluster2:
                age_diff = abs(user["age"] - other["age"])
                if age_diff < 10:
                    return False

                sim = jaccard(user["tags"], other["tags"])
                if sim > 0.7 or sim < 0.2:
                    return False

        return True

    # Merge small clusters until all clusters meet minimum size
    while any(len(cluster) < min_cluster_size for cluster in final_clusters) and len(final_clusters) > 1:
        # Find the smallest cluster
        smallest_idx = min(range(len(final_clusters)), key=lambda i: len(final_clusters[i]))
        smallest_cluster = final_clusters[smallest_idx]

        # If it meets the minimum size, we're done
        if len(smallest_cluster) >= min_cluster_size:
            break

        # Find the best cluster to merge with that maintains common tags
        best_merge_idx = None

        for i, cluster in enumerate(final_clusters):
            if i == smallest_idx:
                continue

            if can_merge_clusters(smallest_cluster, cluster):
                best_merge_idx = i
                break

        # Merge the smallest cluster into the best match
        if best_merge_idx is not None:
            final_clusters[best_merge_idx].extend(smallest_cluster)
            final_clusters.pop(smallest_idx)
        else:
            # If no good merge found, we need to decide:
            # 1. Keep a cluster smaller than the minimum
            # 2. Try to distribute its members
            users_to_distribute = smallest_cluster
            final_clusters.pop(smallest_idx)

            for user in users_to_distribute:
                added = False
                for cluster in final_clusters:
                    would_have_common_tag = bool(get_common_tags(user, cluster))
                    if is_valid(user, cluster) and would_have_common_tag:
                        cluster.append(user)
                        added = True
                        break

                if not added:
                    # No valid cluster found, create a new singleton cluster as last resort
                    orphaned_cluster = [user]
                    final_clusters.append(orphaned_cluster)

    # Format the output to show user IDs and common tags
    result_clusters = []
    for cluster in final_clusters:
        # Extract user IDs
        user_ids = [user["id"] for user in cluster]

        # Find common tags (tags that appear in ALL users in the cluster)
        common_tags = list(set.intersection(*[set(user["tags"]) for user in cluster])) if cluster else []

        result_clusters.append({
            "user_ids": user_ids,
            "common_tags": common_tags,
            "size": len(user_ids)
        })

    return result_clusters


clusters = get_clusters(users, initial_clusters=3, min_cluster_size=4)

for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}:")
    print(f"  User IDs: {cluster['user_ids']}")
    print(f"  Common Tags: {cluster['common_tags']}")
    print(f"  Size: {cluster['size']}")
    print()