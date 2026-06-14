# Add to the top of retrieval_router.py or a shared Pydantic models file
from pydantic import BaseModel, Field
from typing import Optional

class RagFeedbackRequestPayload(BaseModel):
    log_id: int = Field(..., description="The ID of the chat log entry to provide feedback for.")
    feedback_value: int = Field(..., description="Numeric feedback: 1 for thumb up, -1 for thumb down, 0 to clear/neutralize.", ge=-1, le=1)
    feedback_text: Optional[str] = Field(None, description="Optional custom text feedback from the user.", max_length=2000)
    
class RagFeedbackSubmissionResponse(BaseModel):
    status: str
    message: str

class WrappedRagFeedbackResponse(BaseModel): # New wrapper model
    results: RagFeedbackSubmissionResponse 
