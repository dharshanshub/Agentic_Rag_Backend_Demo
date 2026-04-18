import asyncio
import json
import re

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from app.clients.blob_client import BlobStorageClient
from app.core.config import get_settings
from app.exceptions.api_exceptions import LLMError
from app.schemas.chat_schema import ChatResponse, SourceDocument
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AgentService:
    """
    Orchestrates multi-turn conversations with an Azure AI Foundry Agent.

    Uses azure-ai-projects v2 SDK:
      - AIProjectClient.get_openai_client() returns a standard openai.OpenAI client
        scoped to the Foundry project.
      - openai.conversations.create() creates a conversation thread.
      - openai.responses.create() sends a message and gets the agent's grounded response.

    Conversation history is managed by Azure Foundry — the caller only needs
    to pass conversation_id back on each subsequent turn.

    Usage:
        service = AgentService()
        conv_id = await service.create_conversation()
        response = await service.chat("What was decided?", conv_id)
        await service.close()
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._agent_name: str = settings.AZURE_AI_AGENT_NAME
        self._blob_client = BlobStorageClient()

        try:
            self._project = AIProjectClient(
                endpoint=settings.AZURE_AI_PROJECT_ENDPOINT,
                credential=DefaultAzureCredential(),
            )
            # get_openai_client() returns a synchronous openai.OpenAI instance
            # scoped to this Foundry project. All calls are wrapped with
            # asyncio.to_thread so FastAPI's event loop is never blocked.
            self._openai = self._project.get_openai_client()

            logger.info(
                "AgentService initialised | agent='%s'", self._agent_name
            )
        except Exception as exc:
            logger.error("Failed to initialise AgentService: %s", exc)
            raise LLMError(
                f"Could not connect to Azure AI Foundry: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_conversation(self) -> str:
        """
        Start a new conversation thread in Azure AI Foundry.

        Returns:
            conversation_id: Pass this back on every subsequent turn
                             to maintain conversation history.
        """
        try:
            conversation = await asyncio.to_thread(
                self._openai.conversations.create
            )
            logger.info("New conversation created | id=%s", conversation.id)
            return conversation.id
        except Exception as exc:
            logger.error("Failed to create conversation: %s", exc)
            raise LLMError(f"Could not create Foundry conversation: {exc}") from exc

    async def chat(self, question: str, conversation_id: str) -> ChatResponse:
        """
        Send a user message to the Foundry agent and receive a grounded response.

        The agent uses its connected Azure AI Search tool to retrieve relevant
        chunks from the meeting minutes index before generating the answer.

        Args:
            question:        User's natural language question.
            conversation_id: ID from create_conversation() or a prior chat() call.

        Returns:
            ChatResponse with answer, cited sources (with SAS URLs),
            and the conversation_id to continue the thread.
        """
        try:
            logger.info(
                "Sending question to agent | agent='%s' | conversation=%s | question='%s'",
                self._agent_name,
                conversation_id,
                question[:80],
            )

            response = await asyncio.to_thread(
                self._openai.responses.create,
                conversation=conversation_id,
                extra_body={
                    "agent_reference": {
                        "name": self._agent_name,
                        "type": "agent_reference",
                    }
                },
                input=question,
            )

            raw_answer: str = response.output_text or ""
            # Strip Foundry citation markers e.g. 【4:0†source】 — sources are
            # returned separately in the sources array so markers are redundant.
            answer = re.sub(r"【[^】]*】", "", raw_answer).strip()

            # Log raw output during development to understand citation format
            logger.debug("Agent raw output: %s", response.output)

            sources = self._extract_sources(response)

            logger.info(
                "Agent response received | conversation=%s | sources=%d | answer_len=%d",
                conversation_id,
                len(sources),
                len(answer),
            )

            return ChatResponse(
                answer=answer,
                sources=sources,
                conversation_id=conversation_id,
            )

        except LLMError:
            raise
        except Exception as exc:
            logger.error(
                "Agent chat failed | conversation=%s | %s", conversation_id, exc
            )
            raise LLMError(
                f"Azure AI Foundry agent call failed: {exc}"
            ) from exc

    async def close(self) -> None:
        """Release underlying HTTP connections."""
        try:
            self._project.close()
        except Exception:
            pass
        await self._blob_client.close()
        logger.debug("AgentService closed.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_sources(self, response) -> list[SourceDocument]:
        """
        Build SourceDocument list from the Foundry response.

        Two-pass approach:
          Pass 1 — parse the azure_ai_search_call_output item to get full chunk
                   text and date for every retrieved document (keyed by title).
          Pass 2 — parse the message annotations to find which documents the
                   agent actually cited, then look up their content from pass 1.

        Only cited documents appear in the final list so the UI shows only what
        the agent used, not all retrieved candidates.
        """
        try:
            search_docs = self._parse_search_output(response.output or [])
            cited_titles = self._parse_cited_titles(response.output or [])

            sources: list[SourceDocument] = []
            for title in cited_titles:
                doc = search_docs.get(title, {})
                sources.append(
                    SourceDocument(
                        title=title,
                        chunk=doc.get("chunk", ""),
                        date=doc.get("date"),
                        blob_url=self._blob_client.generate_sas_url(title),
                    )
                )
            return sources

        except Exception as exc:
            logger.warning("Could not extract sources from agent response: %s", exc)
            return []

    def _parse_search_output(self, output_items: list) -> dict[str, dict]:
        """
        Parse the azure_ai_search_call_output item and return a dict of
        { title: { chunk: str, date: str | None } } for every retrieved doc.
        """
        for item in output_items:
            if getattr(item, "type", "") != "azure_ai_search_call_output":
                continue
            try:
                data = json.loads(item.output)
                result: dict[str, dict] = {}
                for doc in data.get("documents", []):
                    title = doc.get("title", "")
                    raw_content = doc.get("content", "")
                    if title and title not in result:
                        chunk, date = self._parse_content(raw_content, title)
                        result[title] = {"chunk": chunk, "date": date}
                return result
            except Exception as exc:
                logger.warning("Failed to parse search output JSON: %s", exc)

        return {}

    def _parse_cited_titles(self, output_items: list) -> list[str]:
        """
        Walk the message item's annotations and return an ordered, deduplicated
        list of document titles the agent cited in its answer.
        """
        cited: list[str] = []
        seen: set[str] = set()

        for item in output_items:
            if getattr(item, "type", "") != "message":
                continue
            for content in (getattr(item, "content", []) or []):
                for annotation in (getattr(content, "annotations", []) or []):
                    # url_citation is how Foundry surfaces AI Search citations
                    url_citation = getattr(annotation, "url_citation", None)
                    title = (
                        getattr(url_citation, "title", None)
                        if url_citation
                        else getattr(annotation, "title", None)
                    )
                    if title and title not in seen:
                        seen.add(title)
                        cited.append(title)

        return cited

    def _parse_content(self, raw_content: str, title: str) -> tuple[str, str | None]:
        """
        Clean up the raw content string returned by Azure AI Search.

        The Foundry agent surfaces content in this format:
            {title}\\n{page_num}\\n{chunk text}\\n{YYYY-MM-DD}\\n

        Returns the clean chunk text and the date string (or None).
        """
        lines = raw_content.strip().splitlines()

        # Strip the leading title line
        if lines and lines[0].strip() == title:
            lines = lines[1:]

        # Strip a bare page number line (single integer)
        if lines and lines[0].strip().isdigit():
            lines = lines[1:]

        # Extract date from the last non-empty line if it matches YYYY-MM-DD
        date_str: str | None = None
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        while lines:
            last = lines[-1].strip()
            if not last:
                lines.pop()
            elif date_pattern.match(last):
                date_str = last
                lines.pop()
                break
            else:
                break

        chunk_text = "\n".join(lines).strip()
        return chunk_text, date_str
