import pandas as pd
from sentence_transformers import SentenceTransformer

# Data of users
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

user_df = pd.DataFrame(users_json)

# Function to get embedding from text
def get_embedding(text):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(text)
    return embeddings

# Create a combined text representation for each user
def create_user_text(user):
    tags = ", ".join(user['tags'])  # Convert list of tags into a string
    return f"User ID: {user['id']}, Age: {user['age']}, Interests: {tags}"

# Apply the embedding function to each user in the DataFrame
user_df['embedding'] = user_df.apply(lambda row: get_embedding(create_user_text(row)), axis=1)

# Convert embeddings into a list of lists
embeddings_list = user_df['embedding'].apply(lambda x: x.tolist()).tolist()  # Convert each embedding to list

# Create a new DataFrame with embeddings as columns
embeddings_df = pd.DataFrame(embeddings_list)

# Add original user data to the embeddings DataFrame
final_df = pd.concat([user_df[['id', 'age','tags']], embeddings_df], axis=1)
