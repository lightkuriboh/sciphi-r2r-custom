import logging
import json
import textwrap
import uuid

from datetime import datetime, timezone
from typing import Any, Literal, Optional, List, Dict
from uuid import UUID

from fastapi import Body, Depends, HTTPException, FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from core.base import (
    GenerationConfig,
    Message,
    R2RException,
    SearchMode,
    SearchSettings,
    select_search_filters,
)
from core.base.api.models import (
    WrappedAgentResponse,
    WrappedCompletionResponse,
    WrappedEmbeddingResponse,
    WrappedLLMChatCompletion,
    WrappedRAGResponse,
    WrappedRAGResponseCustom,
    WrappedSearchResponse,
    RAGResponseCustom,
)

from ...abstractions import R2RProviders, R2RServices
from ...config import R2RConfig
from .base_router import BaseRouterV3

from customization.RagFeedbackPayload import RagFeedbackRequestPayload, RagFeedbackSubmissionResponse, WrappedRagFeedbackResponse
from customization.ragai_logging_db import log_chat_interaction, get_next_turn_numbers, update_chat_feedback
from customization.ragai_logging_db import logger as ragai_logger
from customization.RagAiAuthContext import RagAiAuthContext

logger = logging.getLogger(__name__)


def merge_search_settings(
    base: SearchSettings, overrides: SearchSettings
) -> SearchSettings:
    # Convert both to dict
    base_dict = base.model_dump()
    overrides_dict = overrides.model_dump(exclude_unset=True)

    # Update base_dict with values from overrides_dict
    # This ensures that any field set in overrides takes precedence
    for k, v in overrides_dict.items():
        base_dict[k] = v

    # Construct a new SearchSettings from the merged dict
    return SearchSettings(**base_dict)


class RetrievalRouter(BaseRouterV3):
    def __init__(
        self, providers: R2RProviders, services: R2RServices, config: R2RConfig
    ):
        logging.info("Initializing RetrievalRouter")
        super().__init__(providers, services, config)

    def _register_workflows(self):
        pass

    def _prepare_search_settings(
        self,
        auth_user: Any,
        search_mode: SearchMode,
        search_settings: Optional[SearchSettings],
    ) -> SearchSettings:
        """Prepare the effective search settings based on the provided
        search_mode, optional user-overrides in search_settings, and applied
        filters."""
        if search_mode != SearchMode.custom:
            # Start from mode defaults
            effective_settings = SearchSettings.get_default(search_mode.value)
            if search_settings:
                # Merge user-provided overrides
                effective_settings = merge_search_settings(
                    effective_settings, search_settings
                )
        else:
            # Custom mode: use provided settings or defaults
            effective_settings = search_settings or SearchSettings()

        # Apply user-specific filters
        effective_settings.filters = select_search_filters(
            auth_user, effective_settings
        )
        return effective_settings

    def _setup_routes(self):
        @self.router.post(
            "/retrieval/search",
            dependencies=[Depends(self.rate_limit_dependency)],
            summary="Search R2R",
            openapi_extra={
                "x-codeSamples": [
                    {
                        "lang": "Python",
                        "source": textwrap.dedent(
                            """
                            from r2r import R2RClient

                            client = R2RClient()
                            # if using auth, do client.login(...)

                            response = client.retrieval.search(
                                query="What is DeepSeek R1?",
                            )
                            """
                        ),
                    },
                    {
                        "lang": "JavaScript",
                        "source": textwrap.dedent(
                            """
                            const { r2rClient } = require("r2r-js");

                            const client = new r2rClient();
                            // if using auth, do client.login(...)

                            const response = await client.retrieval.search({
                                query: "What is DeepSeek R1?",
                            });
                            """
                        ),
                    },
                    {
                        "lang": "Shell",
                        "source": textwrap.dedent(
                            """
                            # Basic search
                            curl -X POST "http://localhost:7272/v3/retrieval/search" \\
                                -H "Content-Type: application/json" \\
                                -H "Authorization: Bearer YOUR_API_KEY" \\
                                -d '{
                                "query": "What is DeepSeek R1?"
                            }'
                            """
                        ),
                    },
                ]
            },
        )
        @self.base_endpoint
        async def search_app(
            query: str = Body(
                ...,
                description="Search query to find relevant documents",
            ),
            search_mode: SearchMode = Body(
                default=SearchMode.custom,
                description=(
                    "Default value of `custom` allows full control over search settings.\n\n"
                    "Pre-configured search modes:\n"
                    "`basic`: A simple semantic-based search.\n"
                    "`advanced`: A more powerful hybrid search combining semantic and full-text.\n"
                    "`custom`: Full control via `search_settings`.\n\n"
                    "If `filters` or `limit` are provided alongside `basic` or `advanced`, "
                    "they will override the default settings for that mode."
                ),
            ),
            search_settings: Optional[SearchSettings] = Body(
                None,
                description=(
                    "The search configuration object. If `search_mode` is `custom`, "
                    "these settings are used as-is. For `basic` or `advanced`, these settings will override the default mode configuration.\n\n"
                    "Common overrides include `filters` to narrow results and `limit` to control how many results are returned."
                ),
            ),
            auth_user=Depends(self.providers.auth.auth_wrapper()),
        ) -> WrappedSearchResponse:
            """Perform a search query against vector and/or graph-based
            databases.

            **Search Modes:**
            - `basic`: Defaults to semantic search. Simple and easy to use.
            - `advanced`: Combines semantic search with full-text search for more comprehensive results.
            - `custom`: Complete control over how search is performed. Provide a full `SearchSettings` object.

            **Filters:**
            Apply filters directly inside `search_settings.filters`. For example:
            ```json
            {
            "filters": {"document_id": {"$eq": "e43864f5-a36f-548e-aacd-6f8d48b30c7f"}}
            }
            ```
            Supported operators: `$eq`, `$neq`, `$gt`, `$gte`, `$lt`, `$lte`, `$like`, `$ilike`, `$in`, `$nin`.

            **Hybrid Search:**
            Enable hybrid search by setting `use_hybrid_search: true` in search_settings. This combines semantic search with
            keyword-based search for improved results. Configure with `hybrid_settings`:
            ```json
            {
            "use_hybrid_search": true,
            "hybrid_settings": {
                "full_text_weight": 1.0,
                "semantic_weight": 5.0,
                "full_text_limit": 200,
                "rrf_k": 50
            }
            }
            ```

            **Graph-Enhanced Search:**
            Knowledge graph integration is enabled by default. Control with `graph_search_settings`:
            ```json
            {
            "graph_search_settings": {
                "use_graph_search": true,
                "kg_search_type": "local"
            }
            }
            ```

            **Advanced Filtering:**
            Use complex filters to narrow down results by metadata fields or document properties:
            ```json
            {
            "filters": {
                "$and":[
                    {"document_type": {"$eq": "pdf"}},
                    {"metadata.year": {"$gt": 2020}}
                ]
            }
            }
            ```

            **Results:**
            The response includes vector search results and optional graph search results.
            Each result contains the matched text, document ID, and relevance score.

            """
            if not query:
                raise R2RException("Query cannot be empty", 400)
            effective_settings = self._prepare_search_settings(
                auth_user, search_mode, search_settings
            )
            results = await self.services.retrieval.search(
                query=query,
                search_settings=effective_settings,
            )
            return results  # type: ignore

        @self.router.post(
            "/retrieval/rag",
            dependencies=[Depends(self.rate_limit_dependency)],
            summary="RAG Query",
            response_model=None,
            openapi_extra={
                "x-codeSamples": [
                    {
                        "lang": "Python",
                        "source": textwrap.dedent(
                            """
                            from r2r import R2RClient

                            client = R2RClient()
                            # when using auth, do client.login(...)

                            # Basic RAG request
                            response = client.retrieval.rag(
                                query="What is DeepSeek R1?",
                            )
                            """
                        ),
                    },
                    {
                        "lang": "JavaScript",
                        "source": textwrap.dedent(
                            """
                            const { r2rClient } = require("r2r-js");

                            const client = new r2rClient();
                            // when using auth, do client.login(...)

                            // Basic RAG request
                            const response = await client.retrieval.rag({
                                query: "What is DeepSeek R1?",
                            });
                            """
                        ),
                    },
                    {
                        "lang": "Shell",
                        "source": textwrap.dedent(
                            """
                            # Basic RAG request
                            curl -X POST "http://localhost:7272/v3/retrieval/rag" \\
                                -H "Content-Type: application/json" \\
                                -H "Authorization: Bearer YOUR_API_KEY" \\
                                -d '{
                                "query": "What is DeepSeek R1?"
                            }'
                            """
                        ),
                    },
                ]
            },
        )
        @self.base_endpoint
        async def rag_app(
            query: str = Body(...),
            search_mode: SearchMode = Body(
                default=SearchMode.custom,
                description=(
                    "Default value of `custom` allows full control over search settings.\n\n"
                    "Pre-configured search modes:\n"
                    "`basic`: A simple semantic-based search.\n"
                    "`advanced`: A more powerful hybrid search combining semantic and full-text.\n"
                    "`custom`: Full control via `search_settings`.\n\n"
                    "If `filters` or `limit` are provided alongside `basic` or `advanced`, "
                    "they will override the default settings for that mode."
                ),
            ),
            search_settings: Optional[SearchSettings] = Body(
                None,
                description=(
                    "The search configuration object. If `search_mode` is `custom`, "
                    "these settings are used as-is. For `basic` or `advanced`, these settings will override the default mode configuration.\n\n"
                    "Common overrides include `filters` to narrow results and `limit` to control how many results are returned."
                ),
            ),
            rag_generation_config: GenerationConfig = Body(
                default_factory=GenerationConfig,
                description="Configuration for RAG generation",
            ),
            task_prompt: Optional[str] = Body(
                default=None,
                description="Optional custom prompt to override default",
            ),
            include_title_if_available: bool = Body(
                default=False,
                description="Include document titles in responses when available",
            ),
            include_web_search: bool = Body(
                default=False,
                description="Include web search results provided to the LLM.",
            ),
            auth_user=Depends(self.providers.auth.auth_wrapper()),
        ) -> WrappedRAGResponse:
            """Execute a RAG (Retrieval-Augmented Generation) query.

            This endpoint combines search results with language model generation to produce accurate,
            contextually-relevant responses based on your document corpus.

            **Features:**
            - Combines vector search, optional knowledge graph integration, and LLM generation
            - Automatically cites sources with unique citation identifiers
            - Supports both streaming and non-streaming responses
            - Compatible with various LLM providers (OpenAI, Anthropic, etc.)
            - Web search integration for up-to-date information

            **Search Configuration:**
            All search parameters from the search endpoint apply here, including filters, hybrid search, and graph-enhanced search.

            **Generation Configuration:**
            Fine-tune the language model's behavior with `rag_generation_config`:
            ```json
            {
                "model": "openai/gpt-4.1-mini",  // Model to use
                "temperature": 0.7,              // Control randomness (0-1)
                "max_tokens": 1500,              // Maximum output length
                "stream": true                   // Enable token streaming
            }
            ```

            **Model Support:**
            - OpenAI models (default)
            - Anthropic Claude models (requires ANTHROPIC_API_KEY)
            - Local models via Ollama
            - Any provider supported by LiteLLM

            **Streaming Responses:**
            When `stream: true` is set, the endpoint returns Server-Sent Events with the following types:
            - `search_results`: Initial search results from your documents
            - `message`: Partial tokens as they're generated
            - `citation`: Citation metadata when sources are referenced
            - `final_answer`: Complete answer with structured citations

            **Example Response:**
            ```json
            {
            "generated_answer": "DeepSeek-R1 is a model that demonstrates impressive performance...[1]",
            "search_results": { ... },
            "citations": [
                {
                    "id": "cit.123456",
                    "object": "citation",
                    "payload": { ... }
                }
            ]
            }
            ```
            """

            if "model" not in rag_generation_config.model_fields_set:
                rag_generation_config.model = self.config.app.quality_llm

            effective_settings = self._prepare_search_settings(
                auth_user, search_mode, search_settings
            )

            response = await self.services.retrieval.rag(
                query=query,
                search_settings=effective_settings,
                rag_generation_config=rag_generation_config,
                task_prompt=task_prompt,
                include_title_if_available=include_title_if_available,
                include_web_search=include_web_search,
            )

            if rag_generation_config.stream:
                # ========== Streaming path ==========
                async def stream_generator():
                    try:
                        async for chunk in response:
                            if len(chunk) > 1024:
                                for i in range(0, len(chunk), 1024):
                                    yield chunk[i : i + 1024]
                            else:
                                yield chunk
                    except GeneratorExit:
                        # Clean up if needed, then return
                        return

                return StreamingResponse(
                    stream_generator(), media_type="text/event-stream"
                )  # type: ignore
            else:
                return response
            
        # --- RAGAI CUSTOMIZATION: Add Feedback Endpoint ---
        @self.router.post(
            "/ragai/feedback",
            dependencies=[Depends(self.rate_limit_dependency_for_internal_call)], # Reuse rate limiting if appropriate
            summary="Submit Feedback for a Chat Message",
            response_model=WrappedRagFeedbackResponse,
            tags=["RagAI Custom Endpoints"], # Optional: For API docs organization
        )
        @self.base_endpoint # If you use this for common endpoint logic/error handling
        async def feedback_app(
            payload: RagFeedbackRequestPayload,
            ragai_user: RagAiAuthContext=Depends(self.providers.auth.auth_wrapper(is_internal_api_call=True)),
        ) -> WrappedRagFeedbackResponse:
            """
            Allows a client to submit feedback (thumb up/down and optional text) 
            for a specific chat log entry identified by `log_id`.
            """
            customer_api_key = ragai_user.api_key
            if not customer_api_key:
                raise HTTPException(status_code=403, detail=f"Feedback error: Missing API key.")
            ragai_logger.info(f"RAGAI: Received feedback for log_id {payload.log_id} from API key {customer_api_key}. Value: {payload.feedback_value}, Text: '{payload.feedback_text}'")

            success = update_chat_feedback(
                log_id=payload.log_id,
                feedback_value=payload.feedback_value,
                feedback_text=payload.feedback_text
            )

            if not success:
                # update_chat_feedback logs details. Here we determine if it was "not found" vs other error.
                # (The updated update_chat_feedback now returns False if not found or no change,
                # so this check might be simplified, or we could have it raise specific exceptions)
                raise HTTPException(status_code=404, detail=f"Could not update feedback. Log ID {payload.log_id} may not exist or update failed.")

            return RagFeedbackSubmissionResponse(
                status="success", 
                message=f"Feedback for log_id {payload.log_id} processed."
            )

        @self.router.post(
            "/ragai/rag_custom",
            dependencies=[Depends(self.rate_limit_dependency_for_internal_call)],
            summary="RAG Query",
            response_model=None,
            openapi_extra={
                "x-codeSamples": [
                    {
                        "lang": "Python",
                        "source": textwrap.dedent(
                            """
                            from r2r import R2RClient

                            client = R2RClient()
                            # when using auth, do client.login(...)

                            # Basic RAG request
                            response = client.retrieval.rag(
                                query="What is DeepSeek R1?",
                            )
                            """
                        ),
                    },
                    {
                        "lang": "JavaScript",
                        "source": textwrap.dedent(
                            """
                            const { r2rClient } = require("r2r-js");

                            const client = new r2rClient();
                            // when using auth, do client.login(...)

                            // Basic RAG request
                            const response = await client.retrieval.rag({
                                query: "What is DeepSeek R1?",
                            });
                            """
                        ),
                    },
                    {
                        "lang": "Shell",
                        "source": textwrap.dedent(
                            """
                            # Basic RAG request
                            curl -X POST "https://api.sciphi.ai/v3/retrieval/rag" \\
                                -H "Content-Type: application/json" \\
                                -H "Authorization: Bearer YOUR_API_KEY" \\
                                -d '{
                                "query": "What is DeepSeek R1?"
                            }'
                            """
                        ),
                    },
                ]
            },
        )
        @self.base_endpoint
        async def custom_rag_app(
            query: str = Body(...),
            search_mode: SearchMode = Body(
                default=SearchMode.custom,
                description=(
                    "Default value of `custom` allows full control over search settings.\n\n"
                    "Pre-configured search modes:\n"
                    "`basic`: A simple semantic-based search.\n"
                    "`advanced`: A more powerful hybrid search combining semantic and full-text.\n"
                    "`custom`: Full control via `search_settings`.\n\n"
                    "If `filters` or `limit` are provided alongside `basic` or `advanced`, "
                    "they will override the default settings for that mode."
                ),
            ),
            search_settings: Optional[SearchSettings] = Body(
                None,
                description=(
                    "The search configuration object. If `search_mode` is `custom`, "
                    "these settings are used as-is. For `basic` or `advanced`, these settings will override the default mode configuration.\n\n"
                    "Common overrides include `filters` to narrow results and `limit` to control how many results are returned."
                ),
            ),
            rag_generation_config: GenerationConfig = Body(
                default_factory=GenerationConfig,
                description="Configuration for RAG generation",
            ),
            task_prompt: Optional[str] = Body(
                default=None,
                description="Optional custom prompt to override default",
            ),
            include_title_if_available: bool = Body(
                default=False,
                description="Include document titles in responses when available",
            ),
            include_web_search: bool = Body(
                default=False,
                description="Include web search results provided to the LLM.",
            ),
            ragai_user: RagAiAuthContext=Depends(self.providers.auth.auth_wrapper(is_internal_api_call=True)),
            session_id_from_client: Optional[str] = Body(None, alias="sessionId", description="Client-provided session ID for conversation tracking."),
        ) -> Optional[Dict[str, Any]]:
            """Execute a RAG (Retrieval-Augmented Generation) query.

            This endpoint combines search results with language model generation to produce accurate,
            contextually-relevant responses based on your document corpus.

            **Features:**
            - Combines vector search, optional knowledge graph integration, and LLM generation
            - Automatically cites sources with unique citation identifiers
            - Supports both streaming and non-streaming responses
            - Compatible with various LLM providers (OpenAI, Anthropic, etc.)
            - Web search integration for up-to-date information

            **Search Configuration:**
            All search parameters from the search endpoint apply here, including filters, hybrid search, and graph-enhanced search.

            **Generation Configuration:**
            Fine-tune the language model's behavior with `rag_generation_config`:
            ```json
            {
                "model": "openai/gpt-4.1-mini",  // Model to use
                "temperature": 0.7,              // Control randomness (0-1)
                "max_tokens": 1500,              // Maximum output length
                "stream": true                   // Enable token streaming
            }
            ```

            **Model Support:**
            - OpenAI models (default)
            - Anthropic Claude models (requires ANTHROPIC_API_KEY)
            - Local models via Ollama
            - Any provider supported by LiteLLM

            **Streaming Responses:**
            When `stream: true` is set, the endpoint returns Server-Sent Events with the following types:
            - `search_results`: Initial search results from your documents
            - `message`: Partial tokens as they're generated
            - `citation`: Citation metadata when sources are referenced
            - `final_answer`: Complete answer with structured citations

            **Example Response:**
            ```json
            {
            "generated_answer": "DeepSeek-R1 is a model that demonstrates impressive performance...[1]",
            "search_results": { ... },
            "citations": [
                {
                    "id": "cit.123456",
                    "object": "citation",
                    "payload": { ... }
                }
            ]
            }
            ```
            ** Example request for RAGAI customization:
            {
                "query": "What is RAGAI",
                "sessionId": "some-uuid-generated-by-client-for-this-session",
                "search_settings": { ... }, // Optional
                "rag_generation_config": { ... } // Optional
            }
            """
            auth_user = ragai_user.user
            customer_api_key = ragai_user.api_key
            target_model_id_from_header = ragai_user.target_model
            target_rag_config_id_from_header = ragai_user.target_rag_config

            # Validate presence of critical headers (adjust policy as needed)
            if not customer_api_key:
                logger.warning("Header X-Authenticated-Client-Key is missing. Using placeholder for logging.")
                # raise HTTPException(status_code=400, detail="X-Authenticated-Client-Key header is required")
                customer_api_key = "UNKNOWN_API_KEY_FROM_R2R" # Fallback for logging
            
            if not target_model_id_from_header:
                logger.warning(
                    "Header X-Target-Model is missing. Will use model from request body or R2R default."
                )
                # If this header is strictly required by your logic, you might raise an HTTPException here.

            # 2. Determine Session ID and Turn Number
            # R2R might have its own session/conversation management. Try to leverage it.
            # Example: If `auth_user` contains a user ID or session concept, or if payload has conversation_id
            current_session_id = session_id_from_client
            if not current_session_id:
                # If no session_id from client, generate one for this specific exchange.
                # This means each call without a session_id becomes its own "session" in logs.
                current_session_id = str(uuid.uuid4())
                ragai_logger.info(f"RAGAI: No session_id in payload, generated new one for this exchange: {current_session_id}")
            else:
                ragai_logger.info(f"RAGAI: Using session_id from payload: {current_session_id}")
            
            # Turn management needs a robust strategy for multi-turn conversations.
            # For MVP, this is a simplified approach (e.g., each RAG call is one user turn + one assistant turn).
            # TODO: Implement proper turn calculation based on session history.
            # raise NotImplementedError("User session determination!")
            # user_turn_number = 1 # Placeholder
            # assistant_turn_number = user_turn_number + 1
            user_turn_number, assistant_turn_number = get_next_turn_numbers(current_session_id)
            ragai_logger.info(f"RAGAI: Determined turns for session {current_session_id}: User={user_turn_number}, Assistant={assistant_turn_number}")


            # 3. Log User's Turn
            # The `model_id_used` for the user's log entry reflects the model that *will be* targeted for the assistant's response.
            model_to_log_for_user_turn = target_model_id_from_header or \
                                        (rag_generation_config.model if hasattr(rag_generation_config, 'model') and rag_generation_config.model else None) or \
                                        getattr(self.config.app, 'quality_llm', "R2R_DEFAULT_MODEL")

            log_chat_interaction(
                session_id=current_session_id,
                customer_api_key=customer_api_key,
                model_id_used=str(model_to_log_for_user_turn), # Ensure it's a string
                turn=user_turn_number,
                role="user",
                content=query, # User's query from the payload
                retrieved_context=None, # Typically no retrieved_context for the user's own message log
                metadata={"rag_config_id_from_header": target_rag_config_id_from_header} if target_rag_config_id_from_header else None
            )

            # 4. Prepare RAG Generation Config with RagAI Target Model
            # Create a mutable copy of rag_generation_config to avoid modifying the input default factory object directly
            # This depends on GenerationConfig being a Pydantic model or a mutable dict
            
            effective_rag_generation_config: GenerationConfig
            if hasattr(rag_generation_config, 'model_copy') and callable(getattr(rag_generation_config, 'model_copy')): # Pydantic v2 has model_copy
                effective_rag_generation_config = rag_generation_config.model_copy(deep=True)
            elif hasattr(rag_generation_config, 'copy') and callable(getattr(rag_generation_config, 'copy')): # Pydantic v1 had copy
                 effective_rag_generation_config = rag_generation_config.copy(deep=True)
            else: # Fallback for other mutable objects, or if it's already a dict
                effective_rag_generation_config = GenerationConfig(**rag_generation_config.__dict__) # Recreate if unsure

            # Prioritize X-Target-Model header.
            # If not present, keep model from request body's rag_generation_config (if any).
            # If still no model, then R2R's existing logic will apply its default (self.config.app.quality_llm).
            if target_model_id_from_header:
                effective_rag_generation_config.model = target_model_id_from_header
                logger.info(f"RAGAI: Using target model from header: {target_model_id_from_header}")
            elif hasattr(rag_generation_config, 'model') and rag_generation_config.model and "model" in rag_generation_config.model_fields_set:
                # Model was explicitly set in the request body, and no overriding header
                logger.info(f"RAGAI: Using model from request body: {rag_generation_config.model}")
                # effective_rag_generation_config.model is already set from the copy
            else:
                # No header, no model in request body, let R2R's default logic apply or set it explicitly
                effective_rag_generation_config.model = self.config.app.quality_llm
                logger.info(f"RAGAI: No specific model requested; using R2R default quality_llm: {self.config.app.quality_llm}")
            
            actual_model_for_llm_call = str(effective_rag_generation_config.model)

            # --- RAGAI CUSTOMIZATION: Scoped Collection Search Settings ---
            # If the user did not specify any collection filter in the request,
            # but we have a target_rag_config_id_from_header, we inject it.
            # Otherwise, it falls back to searching all documents of the tenant (user).
            has_collection_filter = False
            if search_settings and search_settings.filters:
                for key in search_settings.filters.keys():
                    if "collection_ids" in key:
                        has_collection_filter = True
                        break
            
            if not has_collection_filter and target_rag_config_id_from_header:
                if not search_settings:
                    search_settings = SearchSettings()
                if not search_settings.filters:
                    search_settings.filters = {}
                search_settings.filters["collection_ids"] = {"$overlap": [target_rag_config_id_from_header]}
                ragai_logger.info(f"RAGAI: Injected target collection ID from header/key default: {target_rag_config_id_from_header}")
            # --- End RAGAI CUSTOMIZATION (Preparation Part) ---

            # R2R's existing logic to prepare search settings
            effective_settings = self._prepare_search_settings(
                auth_user, search_mode, search_settings
            )

            # --- Call R2R's core RAG service ---
            # IMPORTANT: Pass the `effective_rag_generation_config` which now includes our target model.
            try:
                # The `response` here is what R2R's service layer returns.
                # It could be a dict, a Pydantic model, or a streaming generator.
                response_from_service = await self.services.retrieval.rag(
                    query=query,
                    search_settings=effective_settings,
                    rag_generation_config=effective_rag_generation_config, # Use our modified config
                    task_prompt=task_prompt,
                    include_title_if_available=include_title_if_available,
                    include_web_search=include_web_search,
                )
            except Exception as e:
                logger.error(f"Error during R2R services.retrieval.rag call: {e}", exc_info=True)
                # Log an error turn if possible/sensible
                log_chat_interaction(
                    session_id=current_session_id,
                    customer_api_key=customer_api_key,
                    model_id_used=actual_model_for_llm_call,
                    turn=assistant_turn_number, # Or a general error turn
                    role="assistant", # Or "system_error"
                    content=f"Error processing RAG request: {str(e)}",
                    metadata={"error": str(e), "rag_config_id_from_header": target_rag_config_id_from_header}
                )
                raise HTTPException(status_code=500, detail=f"Internal error during RAG processing: {str(e)}")

            # --- RAGAI CUSTOMIZATION: Log Assistant's Turn (after getting response) ---
            assistant_response_text = ""
            retrieved_context_for_log = None

            if not effective_rag_generation_config.stream: # Non-streaming case
                # Extract data from r2r_response_data. This depends on its structure.
                # Based on llms.txt R2R response example for RAG:
                # {"generated_answer": "...", "search_results": { ... }, "citations": [...]}
                if isinstance(response_from_service, dict): # RAGResponse
                    assistant_response_text = response_from_service.get("generated_answer", "Error: No generated_answer field in R2R response")
                    retrieved_context_for_log = response_from_service.get("search_results", {}) # Or .search_results
                elif hasattr(response_from_service, 'generated_answer'): # If it's an object
                    assistant_response_text = response_from_service.generated_answer
                    retrieved_context_for_log = getattr(response_from_service, 'search_results', {})
                else:
                    logger.error(f"Unexpected R2R response format (non-streaming): {type(response_from_service)}")
                    assistant_response_text = "Error: Could not parse R2R response."
                
                assistant_log_id_non_stream = log_chat_interaction(
                    session_id=current_session_id,
                    customer_api_key=customer_api_key,
                    model_id_used=actual_model_for_llm_call,
                    turn=assistant_turn_number,
                    role="assistant",
                    content=assistant_response_text,
                    retrieved_context=retrieved_context_for_log,
                    metadata={"rag_config_id_from_header": target_rag_config_id_from_header} if target_rag_config_id_from_header else None
                )
                final_response_payload = response_from_service.model_dump()
                final_response_payload["assistant_log_id"] = assistant_log_id_non_stream
                # Return the original R2R response to the client
                return final_response_payload
                # Use RAGResponseCustom as return type
                # final_response_payload = RAGResponseCustom(
                #     generated_answer=assistant_response_text,
                #     search_results=retrieved_context_for_log,
                #     citations=response_from_service.citations,
                #     metadata=response_from_service.metadata,
                #     completion=response_from_service.completion,
                #     assistant_log_id=assistant_log_id_non_stream
                # )
                # return final_response_payload

            else: # Streaming case
                # For streaming, we need to accumulate the response and log at the end,
                # or log partial chunks if desired (more complex).
                # The example logs after the stream is complete.
                
                # ========== Streaming path ==========
                # The `response_from_service` is the sse_generator from RetrievalService.rag
                
                # --- RAGAI CUSTOMIZATION: Wrap the stream for logging ---
                async def stream_and_log_generator():
                    nonlocal customer_api_key, current_session_id, actual_model_for_llm_call, assistant_turn_number, target_rag_config_id_from_header
                    
                    full_assistant_response_text_parts = []
                    # Initialize with a clear None or empty dict based on what your log_chat_interaction expects
                    retrieved_context_for_logging: Optional[Dict[str, Any]] = None 
                    
                    # Variables to help parse multi-line SSE data fields
                    current_event_type: Optional[str] = None
                    current_event_data_buffer: str = ""

                    try:
                        async for sse_line in response_from_service: # `response_from_service` is R2R's sse_generator
                            # Pass the original SSE line through to the client immediately
                            yield sse_line

                            # Now, try to parse the SSE line for logging purposes
                            sse_line_stripped = sse_line.strip()
                            if not sse_line_stripped: # End of an event
                                if current_event_type and current_event_data_buffer:
                                    try:
                                        event_data_json = json.loads(current_event_data_buffer)
                                        if current_event_type == "message":
                                            delta_obj = event_data_json.get("delta") # `delta` is expected to be the complex object here
                                            if isinstance(delta_obj, dict):
                                                content_list = delta_obj.get("content")
                                                if isinstance(content_list, list):
                                                    for content_item in content_list:
                                                        if isinstance(content_item, dict) and \
                                                        content_item.get("type") == "text" and \
                                                        isinstance(content_item.get("payload"), dict):
                                                            text_val = content_item.get("payload", {}).get("value")
                                                            if isinstance(text_val, str):
                                                                full_assistant_response_text_parts.append(text_val) # Append the actual string value
                                                                ragai_logger.debug(f"RAGAI Stream Log: Appended message delta: '{text_val}'")
                                            elif isinstance(delta_obj, str): # Fallback if R2R sometimes sends a simple string delta
                                                full_assistant_response_text_parts.append(delta_obj)
                                                ragai_logger.debug(f"RAGAI Stream Log: Appended string message delta: '{delta_obj}'")
                                            else:
                                                ragai_logger.warning(f"RAGAI Stream Log: Unexpected delta format in 'message' event: {delta_obj}")

                                        elif current_event_type == "search_results":
                                            # Assuming the 'data' field within the SSE's data payload contains the search results object
                                            captured_search_results_data = event_data_json.get("data")
                                            ragai_logger.info(f"RAGAI Stream Log: Captured search_results event data.")
                                        
                                        elif current_event_type == "final_answer":
                                            final_data = event_data_json.get("data", {})
                                            final_gen_answer = final_data.get("generated_answer", "")
                                            # Capture assistant_log_id if backend sends it with final_answer
                                            # This is an alternative to a separate 'log_info' event.
                                            new_log_id = final_data.get("assistant_log_id")
                                            if new_log_id is not None:
                                                assistant_log_id_for_stream = new_log_id
                                                ragai_logger.info(f"RAGAI Stream Log: Captured assistant_log_id from final_answer: {assistant_log_id_for_stream}")

                                            if final_gen_answer and not "".join(full_assistant_response_text_parts).strip():
                                                # If message events were sparse or didn't capture the full text, use this.
                                                full_assistant_response_text_parts.append(final_gen_answer)
                                                ragai_logger.info(f"RAGAI Stream Log: Appended final_answer to text parts.")
                                        
                                        elif current_event_type == "log_info": # Your custom event for log_id
                                            log_info_data = event_data_json.get("data", {})
                                            new_log_id = log_info_data.get("assistant_log_id")
                                            if new_log_id is not None:
                                                assistant_log_id_for_stream = new_log_id
                                                ragai_logger.info(f"RAGAI Stream Log: Captured assistant_log_id from log_info event: {assistant_log_id_for_stream}")

                                    except json.JSONDecodeError:
                                        ragai_logger.warning(f"RAGAI: Could not JSON decode SSE data for event {current_event_type}: {current_event_data_buffer}")
                                    except Exception as e:
                                        ragai_logger.error(f"RAGAI: Error processing SSE event {current_event_type}: {e}")
                                # Reset for next event
                                current_event_type = None
                                current_event_data_buffer = ""
                                continue # Move to next line

                            if sse_line_stripped.startswith("event:"):
                                current_event_type = sse_line_stripped[len("event:"):].strip()
                            elif sse_line_stripped.startswith("data:"):
                                current_event_data_buffer += sse_line_stripped[len("data:"):].strip()
                            # id: and retry: lines are ignored for logging data content here

                    except GeneratorExit:
                        ragai_logger.info(f"RAGAI: Stream generator for session {current_session_id} exited by client.")
                    except Exception as e:
                        ragai_logger.error(f"RAGAI: Error during RagAI stream wrapping for session {current_session_id}: {e}", exc_info=True)
                        raise # Re-raise to let FastAPI handle it or R2R's error handling
                    finally:
                        assistant_response_text_final = "".join(full_assistant_response_text_parts)
                        if not assistant_response_text_final.strip() and not retrieved_context_for_logging:
                            ragai_logger.warning(f"RAGAI: No content or search results captured from stream for session {current_session_id} to log for assistant.")
                        else:
                            ragai_logger.info(f"RAGAI: Stream finished for session {current_session_id}. Logging accumulated response.")
                            assistant_log_id_for_stream = log_chat_interaction(
                                session_id=current_session_id,
                                customer_api_key=customer_api_key,
                                model_id_used=actual_model_for_llm_call, # Defined earlier in rag_app
                                turn=assistant_turn_number, # Defined earlier
                                role="assistant",
                                content=assistant_response_text_final,
                                retrieved_context=retrieved_context_for_logging,
                                metadata={"rag_config_id_from_header": target_rag_config_id_from_header, "streamed": True} if target_rag_config_id_from_header else {"streamed": True}
                            )
                            yield f"event: log_info\n"
                            yield f"data: {json.dumps({'assistant_log_id': assistant_log_id_for_stream})}\n\n"
                            # Returns using RAGResponseCustom
                            # final_response_payload = RAGResponseCustom(
                            #     generated_answer=response_from_service.generated_answer,
                            #     search_results=response_from_service.search_results, # Pass the AggregateSearchResult object
                            #     citations=response_from_service.citations,
                            #     metadata=response_from_service.metadata,
                            #     completion=response_from_service.completion, # Or generated_answer
                            #     assistant_log_id=assistant_log_id_non_stream
                            # )
                            
                            # # If your endpoint's response_model is set to RAGResponseCustom or Any,
                            # # FastAPI will correctly serialize this Pydantic model.
                            # # If R2R's original rag_app had `response_model=WrappedRAGResponse` and
                            # # WrappedRAGResponse expects a RAGResponse, this should work if RAGResponseCustom is a subclass.
                            # # If you need to return the specific WrappedRAGResponse type:
                            # # return WrappedRAGResponse(results=final_response_payload) # Adjust based on WrappedRAGResponse structure
                            # return final_response_payload # Directly return your Pydantic model
                
                return StreamingResponse(
                    stream_and_log_generator(), media_type="text/event-stream"
                )

        # --- END RAGAI CUSTOMIZATION ---

        @self.router.post(
            "/retrieval/agent",
            dependencies=[Depends(self.rate_limit_dependency)],
            summary="RAG-powered Conversational Agent",
            openapi_extra={
                "x-codeSamples": [
                    {
                        "lang": "Python",
                        "source": textwrap.dedent(
                            """
                            from r2r import (
                                R2RClient,
                                ThinkingEvent,
                                ToolCallEvent,
                                ToolResultEvent,
                                CitationEvent,
                                FinalAnswerEvent,
                                MessageEvent,
                            )

                            client = R2RClient()
                            # when using auth, do client.login(...)

                            # Basic synchronous request
                            response = client.retrieval.agent(
                                message={
                                    "role": "user",
                                    "content": "Do a deep analysis of the philosophical implications of DeepSeek R1"
                                },
                                rag_tools=["web_search", "web_scrape", "search_file_descriptions", "search_file_knowledge", "get_file_content"],
                            )
                            """
                        ),
                    },
                    {
                        "lang": "JavaScript",
                        "source": textwrap.dedent(
                            """
                            const { r2rClient } = require("r2r-js");

                            const client = new r2rClient();
                            // when using auth, do client.login(...)

                            async function main() {
                                // Basic synchronous request
                                const ragResponse = await client.retrieval.agent({
                                    message: {
                                        role: "user",
                                        content: "Do a deep analysis of the philosophical implications of DeepSeek R1"
                                    },
                                    ragTools: ["web_search", "web_scrape", "search_file_descriptions", "search_file_knowledge", "get_file_content"]
                                });
                            }

                            main();
                            """
                        ),
                    },
                    {
                        "lang": "Shell",
                        "source": textwrap.dedent(
                            """
                            # Basic request
                            curl -X POST "http://localhost:7272/v3/retrieval/agent" \\
                                -H "Content-Type: application/json" \\
                                -H "Authorization: Bearer YOUR_API_KEY" \\
                                -d '{
                                "message": {
                                    "role": "user",
                                    "content": "What were the key contributions of Aristotle to logic?"
                                },
                                "search_settings": {
                                    "use_semantic_search": true,
                                    "filters": {"document_id": {"$eq": "e43864f5-a36f-548e-aacd-6f8d48b30c7f"}}
                                },
                                "rag_tools": ["search_file_knowledge", "get_file_content", "web_search"]
                            }'

                            # Advanced analysis with extended thinking
                            curl -X POST "http://localhost:7272/v3/retrieval/agent" \\
                                -H "Content-Type: application/json" \\
                                -H "Authorization: Bearer YOUR_API_KEY" \\
                                -d '{
                                "message": {
                                    "role": "user",
                                    "content": "Do a deep analysis of the philosophical implications of DeepSeek R1"
                                },
                                "search_settings": {"limit": 20},
                                "research_tools": ["rag", "reasoning", "critique", "python_executor"],
                                "rag_generation_config": {
                                    "model": "anthropic/claude-3-7-sonnet-20250219",
                                    "extended_thinking": true,
                                    "thinking_budget": 4096,
                                    "temperature": 1,
                                    "top_p": null,
                                    "max_tokens": 16000,
                                    "stream": False
                                }
                            }'

                            # Conversation continuation
                            curl -X POST "http://localhost:7272/v3/retrieval/agent" \\
                                -H "Content-Type: application/json" \\
                                -H "Authorization: Bearer YOUR_API_KEY" \\
                                -d '{
                                "message": {
                                    "role": "user",
                                    "content": "How does it compare to other reasoning models?"
                                },
                                "conversation_id": "YOUR_CONVERSATION_ID"
                            }'
                            """
                        ),
                    },
                ]
            },
        )
        @self.base_endpoint
        async def agent_app(
            message: Optional[Message] = Body(
                None,
                description="Current message to process",
            ),
            messages: Optional[list[Message]] = Body(
                None,
                deprecated=True,
                description="List of messages (deprecated, use message instead)",
            ),
            search_mode: SearchMode = Body(
                default=SearchMode.custom,
                description="Pre-configured search modes: basic, advanced, or custom.",
            ),
            search_settings: Optional[SearchSettings] = Body(
                None,
                description="The search configuration object for retrieving context.",
            ),
            # Generation configurations
            rag_generation_config: GenerationConfig = Body(
                default_factory=GenerationConfig,
                description="Configuration for RAG generation in 'rag' mode",
            ),
            research_generation_config: Optional[GenerationConfig] = Body(
                None,
                description="Configuration for generation in 'research' mode. If not provided but mode='research', rag_generation_config will be used with appropriate model overrides.",
            ),
            # Tool configurations
            # FIXME: We need a more generic way to handle this
            rag_tools: Optional[
                list[
                    Literal[
                        "web_search",
                        "web_scrape",
                        "search_file_descriptions",
                        "search_file_knowledge",
                        "get_file_content",
                    ]
                ]
            ] = Body(
                None,
                description="List of tools to enable for RAG mode. Available tools: search_file_knowledge, get_file_content, web_search, web_scrape, search_file_descriptions",
            ),
            # FIXME: We need a more generic way to handle this
            research_tools: Optional[
                list[
                    Literal["rag", "reasoning", "critique", "python_executor"]
                ]
            ] = Body(
                None,
                description="List of tools to enable for Research mode. Available tools: rag, reasoning, critique, python_executor",
            ),
            # Backward compatibility
            task_prompt: Optional[str] = Body(
                default=None,
                description="Optional custom prompt to override default",
            ),
            # Backward compatibility
            include_title_if_available: bool = Body(
                default=True,
                description="Pass document titles from search results into the LLM context window.",
            ),
            conversation_id: Optional[UUID] = Body(
                default=None,
                description="ID of the conversation",
            ),
            max_tool_context_length: Optional[int] = Body(
                default=32_768,
                description="Maximum length of returned tool context",
            ),
            use_system_context: Optional[bool] = Body(
                default=True,
                description="Use extended prompt for generation",
            ),
            # FIXME: We need a more generic way to handle this
            mode: Optional[Literal["rag", "research"]] = Body(
                default="rag",
                description="Mode to use for generation: 'rag' for standard retrieval or 'research' for deep analysis with reasoning capabilities",
            ),
            needs_initial_conversation_name: Optional[bool] = Body(
                default=None,
                description="If true, the system will automatically assign a conversation name if not already specified previously.",
            ),
            auth_user=Depends(self.providers.auth.auth_wrapper()),
        ) -> WrappedAgentResponse:
            """
            Engage with an intelligent agent for information retrieval, analysis, and research.

            This endpoint offers two operating modes:
            - **RAG mode**: Standard retrieval-augmented generation for answering questions based on knowledge base
            - **Research mode**: Advanced capabilities for deep analysis, reasoning, and computation

            ### RAG Mode (Default)

            The RAG mode provides fast, knowledge-based responses using:
            - Semantic and hybrid search capabilities
            - Document-level and chunk-level content retrieval
            - Optional web search integration
            - Source citation and evidence-based responses

            ### Research Mode

            The Research mode builds on RAG capabilities and adds:
            - A dedicated reasoning system for complex problem-solving
            - Critique capabilities to identify potential biases or logical fallacies
            - Python execution for computational analysis
            - Multi-step reasoning for deeper exploration of topics

            ### Available Tools

            **RAG Tools:**
            - `search_file_knowledge`: Semantic/hybrid search on your ingested documents
            - `search_file_descriptions`: Search over file-level metadata
            - `content`: Fetch entire documents or chunk structures
            - `web_search`: Query external search APIs for up-to-date information
            - `web_scrape`: Scrape and extract content from specific web pages

            **Research Tools:**
            - `rag`: Leverage the underlying RAG agent for information retrieval
            - `reasoning`: Call a dedicated model for complex analytical thinking
            - `critique`: Analyze conversation history to identify flaws and biases
            - `python_executor`: Execute Python code for complex calculations and analysis

            ### Streaming Output

            When streaming is enabled, the agent produces different event types:
            - `thinking`: Shows the model's step-by-step reasoning (when extended_thinking=true)
            - `tool_call`: Shows when the agent invokes a tool
            - `tool_result`: Shows the result of a tool call
            - `citation`: Indicates when a citation is added to the response
            - `message`: Streams partial tokens of the response
            - `final_answer`: Contains the complete generated answer and structured citations

            ### Conversations

            Maintain context across multiple turns by including `conversation_id` in each request.
            After your first call, store the returned `conversation_id` and include it in subsequent calls.
            If no conversation name has already been set for the conversation, the system will automatically assign one.

            """
            # Handle model selection based on mode
            if "model" not in rag_generation_config.model_fields_set:
                if mode == "rag":
                    rag_generation_config.model = self.config.app.quality_llm
                elif mode == "research":
                    rag_generation_config.model = self.config.app.planning_llm

            # Prepare search settings
            effective_settings = self._prepare_search_settings(
                auth_user, search_mode, search_settings
            )

            # Determine effective generation config
            effective_generation_config = rag_generation_config
            if mode == "research" and research_generation_config:
                effective_generation_config = research_generation_config

            try:
                response = await self.services.retrieval.agent(
                    message=message,
                    messages=messages,
                    search_settings=effective_settings,
                    rag_generation_config=rag_generation_config,
                    research_generation_config=research_generation_config,
                    task_prompt=task_prompt,
                    include_title_if_available=include_title_if_available,
                    max_tool_context_length=max_tool_context_length or 32_768,
                    conversation_id=(
                        str(conversation_id) if conversation_id else None  # type: ignore
                    ),
                    use_system_context=use_system_context
                    if use_system_context is not None
                    else True,
                    rag_tools=rag_tools,  # type: ignore
                    research_tools=research_tools,  # type: ignore
                    mode=mode,
                    needs_initial_conversation_name=needs_initial_conversation_name,
                )

                if effective_generation_config.stream:

                    async def stream_generator():
                        try:
                            async for chunk in response:
                                if len(chunk) > 1024:
                                    for i in range(0, len(chunk), 1024):
                                        yield chunk[i : i + 1024]
                                else:
                                    yield chunk
                        except GeneratorExit:
                            # Clean up if needed, then return
                            return

                    return StreamingResponse(  # type: ignore
                        stream_generator(), media_type="text/event-stream"
                    )
                else:
                    return response
            except Exception as e:
                logger.error(f"Error in agent_app: {e}")
                raise R2RException(str(e), 500) from e

        @self.router.post(
            "/retrieval/completion",
            dependencies=[Depends(self.rate_limit_dependency)],
            summary="Generate Message Completions",
            openapi_extra={
                "x-codeSamples": [
                    {
                        "lang": "Python",
                        "source": textwrap.dedent(
                            """
                            from r2r import R2RClient

                            client = R2RClient()
                            # when using auth, do client.login(...)

                            response = client.completion(
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant."},
                                    {"role": "user", "content": "What is the capital of France?"},
                                    {"role": "assistant", "content": "The capital of France is Paris."},
                                    {"role": "user", "content": "What about Italy?"}
                                ],
                                generation_config={
                                    "model": "openai/gpt-4.1-mini",
                                    "temperature": 0.7,
                                    "max_tokens": 150,
                                    "stream": False
                                }
                            )
                            """
                        ),
                    },
                    {
                        "lang": "JavaScript",
                        "source": textwrap.dedent(
                            """
                            const { r2rClient } = require("r2r-js");

                            const client = new r2rClient();
                            // when using auth, do client.login(...)

                            async function main() {
                                const response = await client.completion({
                                    messages: [
                                        { role: "system", content: "You are a helpful assistant." },
                                        { role: "user", content: "What is the capital of France?" },
                                        { role: "assistant", content: "The capital of France is Paris." },
                                        { role: "user", content: "What about Italy?" }
                                    ],
                                    generationConfig: {
                                        model: "openai/gpt-4.1-mini",
                                        temperature: 0.7,
                                        maxTokens: 150,
                                        stream: false
                                    }
                                });
                            }

                            main();
                            """
                        ),
                    },
                    {
                        "lang": "Shell",
                        "source": textwrap.dedent(
                            """
                            curl -X POST "http://localhost:7272/v3/retrieval/completion" \\
                                -H "Content-Type: application/json" \\
                                -H "Authorization: Bearer YOUR_API_KEY" \\
                                -d '{
                                "messages": [
                                    {"role": "system", "content": "You are a helpful assistant."},
                                    {"role": "user", "content": "What is the capital of France?"},
                                    {"role": "assistant", "content": "The capital of France is Paris."},
                                    {"role": "user", "content": "What about Italy?"}
                                ],
                                "generation_config": {
                                    "model": "openai/gpt-4.1-mini",
                                    "temperature": 0.7,
                                    "max_tokens": 150,
                                    "stream": false
                                }
                                }'
                            """
                        ),
                    },
                ]
            },
        )
        @self.base_endpoint
        async def completion(
            messages: list[Message] = Body(
                ...,
                description="List of messages to generate completion for",
                example=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {
                        "role": "user",
                        "content": "What is the capital of France?",
                    },
                    {
                        "role": "assistant",
                        "content": "The capital of France is Paris.",
                    },
                    {"role": "user", "content": "What about Italy?"},
                ],
            ),
            generation_config: GenerationConfig = Body(
                default_factory=GenerationConfig,
                description="Configuration for text generation",
                example={
                    "model": "openai/gpt-4.1-mini",
                    "temperature": 0.7,
                    "max_tokens": 150,
                    "stream": False,
                },
            ),
            auth_user=Depends(self.providers.auth.auth_wrapper()),
            response_model=WrappedCompletionResponse,
        ) -> WrappedLLMChatCompletion:
            """Generate completions for a list of messages.

            This endpoint uses the language model to generate completions for
            the provided messages. The generation process can be customized
            using the generation_config parameter.

            The messages list should contain alternating user and assistant
            messages, with an optional system message at the start. Each
            message should have a 'role' and 'content'.
            """

            return await self.services.retrieval.completion(
                messages=messages,  # type: ignore
                generation_config=generation_config,
            )

        @self.router.post(
            "/retrieval/embedding",
            dependencies=[Depends(self.rate_limit_dependency)],
            summary="Generate Embeddings",
            openapi_extra={
                "x-codeSamples": [
                    {
                        "lang": "Python",
                        "source": textwrap.dedent(
                            """
                            from r2r import R2RClient

                            client = R2RClient()
                            # when using auth, do client.login(...)

                            result = client.retrieval.embedding(
                                text="What is DeepSeek R1?",
                            )
                            """
                        ),
                    },
                    {
                        "lang": "JavaScript",
                        "source": textwrap.dedent(
                            """
                            const { r2rClient } = require("r2r-js");

                            const client = new r2rClient();
                            // when using auth, do client.login(...)

                            async function main() {
                                const response = await client.retrieval.embedding({
                                    text: "What is DeepSeek R1?",
                                });
                            }

                            main();
                            """
                        ),
                    },
                    {
                        "lang": "Shell",
                        "source": textwrap.dedent(
                            """
                            curl -X POST "http://localhost:7272/v3/retrieval/embedding" \\
                                -H "Content-Type: application/json" \\
                                -H "Authorization: Bearer YOUR_API_KEY" \\
                                -d '{
                                "text": "What is DeepSeek R1?",
                                }'
                            """
                        ),
                    },
                ]
            },
        )
        @self.base_endpoint
        async def embedding(
            text: str = Body(
                ...,
                description="Text to generate embeddings for",
            ),
            auth_user=Depends(self.providers.auth.auth_wrapper()),
        ) -> WrappedEmbeddingResponse:
            """Generate embeddings for the provided text using the specified
            model.

            This endpoint uses the language model to generate embeddings for
            the provided text. The model parameter specifies the model to use
            for generating embeddings.
            """

            return await self.services.retrieval.embedding(
                text=text,
            )
