from pydantic import BaseModel
from typing import List, Tuple, Optional
# This file will be responsible for receiving and validating data
# Pydantic models
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Tuple[str, str]]] = []
    session_id: str = "unknown"


class LeadSubmitRequest(BaseModel):
    session_id: str
    phone_number: str
    question_count: int = 0
    asked_about_price: bool = False
    is_registered: Optional[str] = None
    city: Optional[str] = None


class LeadUpdateRequest(BaseModel):
    admin_note: Optional[str] = None
    lead_status: Optional[str] = None
    is_contacted: Optional[bool] = None


class BulkLeadContactedRequest(BaseModel):
    lead_ids: List[int]
    is_contacted: bool

