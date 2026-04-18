from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat query from the user."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The natural language question to answer from meeting minutes.",
    )
    conversation_id: str | None = Field(
        default=None,
        description=(
            "Conversation ID for multi-turn chat. "
            "Omit or set null to start a new conversation. "
            "Pass the value returned in the previous response to continue."
        ),
    )


class SourceDocument(BaseModel):
    """A single cited source document from the agent's grounded response."""

    title: str = Field(description="Filename of the source document.")
    chunk: str = Field(description="The cited text passage from the document.")
    date: str | None = Field(
        default=None,
        description="Meeting date extracted from the document (YYYY-MM-DD), if available.",
    )
    blob_url: str = Field(
        description="Time-limited SAS URL to open the source PDF directly in a browser."
    )


class ChatResponse(BaseModel):
    """AI-generated answer with supporting source documents and conversation context."""

    answer: str = Field(description="The AI-generated answer based on retrieved context.")
    sources: list[SourceDocument] = Field(
        description="Source documents cited by the agent to generate the answer."
    )
    conversation_id: str = Field(
        description=(
            "Pass this value in the next request to continue the conversation. "
            "Azure AI Foundry retains the full message history under this ID."
        )
    )
