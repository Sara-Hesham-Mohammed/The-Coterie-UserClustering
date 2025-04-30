import json
import pandas as pd
from sentence_transformers import SentenceTransformer

# Create a combined text representation for each user
def create_user_text(user):
    tags = ", ".join(user['tags'])  # Convert list of tags into a string
    return f"User ID: {user['id']}, Age: {user['age']}, Interests: {tags}"

# Function to get embedding from text
def get_embedding(text):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(text)
    return embeddings
