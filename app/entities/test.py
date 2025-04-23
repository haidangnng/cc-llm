from pydantic import BaseModel


class TestRequest(BaseModel):
    course_id: str
    topic: str
