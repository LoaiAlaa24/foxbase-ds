from langchain_core.pydantic_v1 import BaseModel, Field

class PagesRange(BaseModel):
    pages_range: list = Field(description="[start_page, last_page]")
