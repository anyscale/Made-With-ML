from typing import Optional

from fastapi import Query
from pydantic import BaseModel


class Project(BaseModel):
    title: Optional[str] = Query(default="", min_length=1)
    description: Optional[str] = Query(default="", min_length=1)

    class Config:
        schema_extra = {
            "example": {
                "title": "Transfer learning with transformers",
                "description": "Using transformers for transfer learning on text classification tasks.",
            }
        }
