import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from Models.User_Model import UserDTO
from typing import List
from Clusters.GaussianClusters import get_clusters


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
    return {"message": "This is the python backend for the clustering(group formation)."}

@app.post("/get-group/")
async def form_groups(request: dict):
    # Validate that 'users' key exists
    if 'users' not in request:
        raise HTTPException(
            status_code=422,
            detail="Request must contain 'users' key with user data"
        )

    users_data = request['users']

    # Validate that users_data is a list and not empty
    if not isinstance(users_data, list) or not users_data:
        raise HTTPException(
            status_code=422,
            detail="Users data must be a non-empty list"
        )

    # Validate each user
    validated_users = []

    for i, user_data in enumerate(users_data):
        try:
            # Convert user JSON data to string for validation
            user_json = json.dumps(user_data)

            # Convert the JSON to UserDTO for validation
            user_dto = UserDTO.model_validate_json(user_json)

            if not user_dto:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid user data at index {i} - missing required fields: id, interest_embedding, location_lang_embedding"
                )

            validated_users.append(user_dto.model_dump())

        except Exception as e:
            print(f"Validation error for user at index {i}: {e}")
            raise HTTPException(
                status_code=422,
                detail=f"Invalid data for user at index {i}: {str(e)}"
            )

    print(f"Successfully validated {len(validated_users)} users")

    # Pass the users data to your clustering function
    groups = get_clusters(validated_users)
    return groups  # Convert numpy array to list for JSON serialization
