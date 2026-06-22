from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentRecord(BaseModel):
    doc_id: str
    company: str = "SAP"
    title: str
    source: str
    source_type: str
    url: str
    published: Optional[str] = None
    author: Optional[str] = None
    content: str

    category: str = "general"
    subcategory: Optional[str] = None
    competitor: Optional[str] = None
    sentiment: Optional[str] = None
    language: str = "en"

    tags: List[str] = Field(default_factory=list)
    collected_at: Optional[str] = None
    content_hash: Optional[str] = None