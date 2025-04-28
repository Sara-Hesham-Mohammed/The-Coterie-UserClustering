import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import false_discovery_control

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

df = pd.DataFrame(users_json)

# Function to calculate Jaccard similarity
def jaccard(set1, set2):
    # TO ENSURE THAT THE INPUTS ARE SETS
    if type(set1,set2) != set:
        set1 = set(set1)
        set2 = set(set2)
    # returns the decimal % of similarity between the two sets
    return len(set1.intersection(set2)) / len(set1.union(set2))

# Helper function to find common tags between users in a cluster
def get_common_tags(cluster):
    for user in cluster:
        user_tags = set(user["tags"])
        if not cluster: # If cluster is empty, return user tags
            print(f"Cluster is empty, returning user tags: {user_tags}")  # Debugging line
            return user_tags
        else:
            cluster_common_tags = set.intersection(*[set(u["tags"]) for u in cluster])
            return cluster_common_tags

def is_valid(user, cluster):
    common_tags = get_common_tags(cluster)
    age_constraint = True
    jaccard_constraint = True

    # If cluster is empty, it's valid
    if not cluster:
        return True

    # Check age group constraint
    for other_user in cluster:
        if abs(user["age"] - other_user["age"]) < 10:  # Example age constraint
            age_constraint = False
            print(f"Age constraint failed between user {user['id']} and user {other_user['id']}")
            break

    # Check tag similarity constraints
    for other_user in cluster:
        sim = jaccard(user["tags"], other_user["tags"])
        if sim <= 0.2 or sim >= 0.7:  # Constraint 2: tag similarity constraints
            print(f"Tag similarity constraint failed between user {user['id']} and user {other_user['id']}")  # Debugging line
            return False
        elif 0.2 < sim < 0.7:
            jaccard_constraint = True
            print(f"Tag similarity constraint passed between user {user['id']} and user {other_user['id']}")
        else:
            print(f"Something wrong in conditional statement")

    # Ensure at least one common tag with the cluster
    if not common_tags:
        print(f"No common tags found between user {user['id']} and cluster")
        return False

    return age_constraint and jaccard_constraint

def create_clusters(users, max_size=5):
    clusters = [] # list of ALL clusters => hashmaps (dictionaries apparently) "list of common tags" : "list of people"
    cluster = [] # list of people in a single cluster

    tags = get_common_tags(cluster)
    cluster_dict = {tags: cluster} # hashmap of common tags to people
    clusters.append(cluster_dict)

    for user in users:
        if is_valid(user, cluster):
            cluster.append(user)
        if len(cluster) >= max_size:
            break

    return clusters


# create_clusters(users_json, 5)