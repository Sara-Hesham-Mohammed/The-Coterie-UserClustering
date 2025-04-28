from kmodes.kprototypes import KPrototypes
import numpy as np
from sklearn.preprocessing import LabelEncoder

users_json = [
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

# Prepare the data
data = []
tags_all = []  # To keep track of all tags for encoding

for user in users_json:
    # Flatten each user into a row with ID, age, and tags
    row = [user["id"], user["age"]]  # Start with ID and age
    data.append(row)
    tags_all.extend(user["tags"])  # Collect all tags to encode

# Remove duplicate tags
tags_all = list(set(tags_all))

# Create a label encoder for tags
le = LabelEncoder()
le.fit(tags_all)

# Convert tags into numerical representations
for i, user in enumerate(users_json):
    encoded_tags = le.transform(user["tags"])
    data[i].extend(encoded_tags)  # Add encoded tags to the row

# Convert to a NumPy array
X = np.array(data, dtype=object)

# Define the model
kproto = KPrototypes(n_clusters=2, init='Cao', verbose=2)

# Fit and predict cluster labels
clusters = kproto.fit_predict(X, categorical=[1] + list(range(2, X.shape[1])))

# Output the cluster labels for each user
print("Users and their cluster labels:")
for i, user in enumerate(users_json):
    print(f"User {user['id']} (Age: {user['age']}, Tags: {user['tags']}) is in cluster {clusters[i]}")

# Optionally, group users by cluster
clustered_users = {0: [], 1: []}
for i, label in enumerate(clusters):
    clustered_users[label].append(users_json[i])

# Display the users in each cluster
for cluster_id, users_in_cluster in clustered_users.items():
    print(f"\nCluster {cluster_id}:")
    for user in users_in_cluster:
        print(f"User {user['id']} (Age: {user['age']}, Tags: {user['tags']})")
