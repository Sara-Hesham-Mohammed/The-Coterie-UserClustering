import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Models.User_Model import User
from typing import List
from Clusters.GaussianClusters import get_clusters
from preprocessing import preproc

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
    proc_users_dict = {}
    for user in users:
        proc_users_dict[user] = preproc(user)

    groups = get_clusters(proc_users_dict)
    return groups
