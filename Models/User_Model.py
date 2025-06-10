from pydantic import BaseModel
from typing import List, Optional

class User(BaseModel):
    id: int
    tags: List[str]
    country: str
    languages: List[str]
    preferred_group_size: Optional[str] = "any"  # can be "small", "medium", "large", or "any"
