import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from API.User_Model import User
from typing import List
from Clusters.GaussianClusters import dimensionality_reduction, get_clusters

app = FastAPI()
#to be able to send requests using html file?
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get("/")
async def root():
    return {"message": "This is the python backend for the embeddings and clustering(group formation)."}

@app.post("/get-group/")
async def form_groups(users: List[User]):
    user_df = pd.DataFrame(users)
    embeddings_df = user_df['embedding']
    reduced_df = dimensionality_reduction(embeddings_df)
    # Perform GMM clustering (soft clustering)
    # pass the number of clusters as a parameter, default is 5
    groups = get_clusters(reduced_df)
    return groups
