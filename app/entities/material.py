from pydantic import BaseModel
from typing import Optional


class MaterialChunk(BaseModel):
    course_id: str
    content: str


class MaterialQuery(BaseModel):
    course_id: str
    query: str
    top_k: Optional[int] = 5
