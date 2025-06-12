from pydantic import BaseModel
from typing import List, Optional

class UserDTO(BaseModel):
    id: int
    interest_embedding: List[float]
    location_lang_embedding: List[float]
