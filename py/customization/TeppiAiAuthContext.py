
from pydantic import BaseModel, Field
from core.base.api.models import User
from typing import  Optional

class RagAiAuthContext(BaseModel):
    user: User = Field(..., description="The authenticated R2R user object.")
    api_key: Optional[str] = Field(None, description="The RagAI API key used for the request.")
    target_model: Optional[str] = Field(None, description="The target LLM model ID for this request.")
    target_rag_config: Optional[str] = Field(None, description="The target RAG configuration ID, if any.")
