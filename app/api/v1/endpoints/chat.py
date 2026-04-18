from fastapi import APIRouter, HTTPException, status

from app.exceptions.api_exceptions import LLMError
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.services.agent_service import AgentService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/query",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Answer a question using meeting minutes via Azure AI Foundry Agent",
)
async def chat_query(request: ChatRequest) -> ChatResponse:
    """
    Multi-turn RAG endpoint powered by Azure AI Foundry Agent.

    The agent uses its connected Azure AI Search tool to retrieve relevant
    chunks from the meeting minutes index, then generates a grounded answer.

    Pipeline:
    1. Create or resume a Foundry conversation thread.
    2. Send the question to the agent.
    3. Agent retrieves from AI Search + generates answer internally.
    4. Parse cited source documents and attach SAS URLs.
    5. Return answer, sources, and conversation_id.

    Request body:
    - ``question``        (str):       Natural language question.
    - ``conversation_id`` (str|null):  Omit or null to start a new conversation.
                                       Pass the returned value to continue.

    Response:
    - ``answer``          (str):  AI-generated answer grounded in meeting minutes.
    - ``sources``         (list): Cited source documents with SAS URLs.
    - ``conversation_id`` (str):  Pass in subsequent requests to maintain history.
    """
    agent_service = AgentService()

    try:
        # ------------------------------------------------------------------
        # Start a new conversation or continue an existing one.
        # Foundry stores the full thread history — no local state needed.
        # ------------------------------------------------------------------
        conversation_id = request.conversation_id
        if not conversation_id:  # handles both None and empty string ""
            conversation_id = await agent_service.create_conversation()
            logger.info("Started new conversation | id=%s", conversation_id)
        else:
            logger.info("Continuing conversation | id=%s", conversation_id)

        # ------------------------------------------------------------------
        # Send question → agent retrieves from AI Search + generates answer
        # ------------------------------------------------------------------
        response = await agent_service.chat(
            question=request.question,
            conversation_id=conversation_id,
        )

        return response

    except LLMError as exc:
        logger.error("Agent service failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        )

    finally:
        await agent_service.close()
