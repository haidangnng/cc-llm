from pydantic import BaseModel
from typing import Optional, List


class TestRequest(BaseModel):
    course_id: str
    topic: str


class GenerateQuestionRequest(BaseModel):
    materialIds: List[int]
    count: Optional[int] = 5


class GenerateAnswerRequest(BaseModel):
    materialIds: List[int]
    question: str
    count: Optional[int] = 4
