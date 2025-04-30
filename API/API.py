import pickle

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from API.User_Model import User
from typing import List
from Clusters import user_embedding
from Clusters.GaussianClusters import dimensionality_reduction, get_clusters
from Clusters.user_embedding import create_user_text


def get_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
        return model

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
    return {"message": "This is the python backend for the recommendation system and group formation."}

@app.post("/recommendation/")
async def get_recommendation(user):
    # PASS THE USER STUFF IN THE PARAMS
    # call the graph rec model here?
    model = get_model("models/recommendation.pth")
    user = 'smth'
    #prediction = model.predict(user)
    prediction = ['x','y','z']
    return {"Prediction": f"Get user info and return the recommendations for that user, prediction is {prediction[1]}"}

@app.post("/store-embedding/")
async def store_embedding(user: User):
    usr_text = create_user_text(user);
    embedding = user_embedding.get_embedding(usr_text)
    #this is what will get sent to the database service API (node.js)
    return {"embedding": embedding.tolist()}

@app.post("/get-group/")
async def form_groups(users: List[User]):
    user_df = pd.DataFrame(users)
    reduced_df = dimensionality_reduction(user_df)
    # Perform GMM clustering (soft clustering)
    groups = get_clusters(reduced_df)
    return groups