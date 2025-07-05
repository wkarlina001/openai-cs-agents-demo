from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import uuid4
import time
import logging
from agents import RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, ModelProvider, Model
from src.metrics_logger import log_metric # Import the new logging function

from main import (
    triage_agent,
    faq_agent,
    recommendation_agent,
    create_initial_context,
)

from agents import (
    Runner,
    ItemHelpers,
    MessageOutputItem,
    HandoffOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
    InputGuardrailTripwireTriggered,
    Handoff,
)
# from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS configuration (adjust as needed for deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Models
# =========================

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str

class MessageResponse(BaseModel):
    content: str
    agent: str

class AgentEvent(BaseModel):
    id: str
    type: str
    agent: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

class GuardrailCheck(BaseModel):
    id: str
    name: str
    input: str
    reasoning: str
    passed: bool
    timestamp: float

class ChatResponse(BaseModel):
    conversation_id: str
    current_agent: str
    messages: List[MessageResponse]
    events: List[AgentEvent]
    context: Dict[str, Any]
    agents: List[Dict[str, Any]]
    guardrails: List[GuardrailCheck] = []

# =========================
# In-memory store for conversation state
# =========================

class ConversationStore:
    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        pass

    def save(self, conversation_id: str, state: Dict[str, Any]):
        pass

class InMemoryConversationStore(ConversationStore):
    _conversations: Dict[str, Dict[str, Any]] = {}

    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._conversations.get(conversation_id)

    def save(self, conversation_id: str, state: Dict[str, Any]):
        self._conversations[conversation_id] = state

# TODO: when deploying this app in scale, switch to your own production-ready implementation
conversation_store = InMemoryConversationStore()
logger.warning("Using InMemoryConversationStore. Switch to a persistent store for production.") #

# =========================
# Helpers
# =========================

def _get_agent_by_name(name: str):
    """Return the agent object by name."""
    agents = {
        triage_agent.name: triage_agent,
        faq_agent.name: faq_agent,
        recommendation_agent.name: recommendation_agent
    }
    return agents.get(name, triage_agent)

def _get_guardrail_name(g) -> str:
    """Extract a friendly guardrail name."""
    name_attr = getattr(g, "name", None)
    if isinstance(name_attr, str) and name_attr:
        return name_attr
    guard_fn = getattr(g, "guardrail_function", None)
    if guard_fn is not None and hasattr(guard_fn, "__name__"):
        return guard_fn.__name__.replace("_", " ").title()
    fn_name = getattr(g, "__name__", None)
    if isinstance(fn_name, str) and fn_name:
        return fn_name.replace("_", " ").title()
    return str(g)

def _build_agents_list() -> List[Dict[str, Any]]:
    """Build a list of all available agents and their metadata."""
    def make_agent_dict(agent):
        return {
            "name": agent.name,
            "description": getattr(agent, "handoff_description", ""),
            "handoffs": [getattr(h, "agent_name", getattr(h, "name", "")) for h in getattr(agent, "handoffs", [])],
            "tools": [getattr(t, "name", getattr(t, "__name__", "")) for t in getattr(agent, "tools", [])],
            "input_guardrails": [_get_guardrail_name(g) for g in getattr(agent, "input_guardrails", [])],
        }
    return [
        make_agent_dict(triage_agent),
        make_agent_dict(faq_agent),
        make_agent_dict(recommendation_agent)
    ]

# =========================
# Main Chat Endpoint
# =========================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Main chat endpoint for agent orchestration.
    Handles conversation state, agent routing, and guardrail checks.
    """
    # Initialize or retrieve conversation state
    is_new = not req.conversation_id or conversation_store.get(req.conversation_id) is None
    if is_new:
        conversation_id: str = uuid4().hex
        ctx = create_initial_context()
        current_agent_name = triage_agent.name
        state: Dict[str, Any] = {
            "input_items": [],
            "context": ctx,
            "current_agent": current_agent_name,
        }
        if req.message.strip() == "":
            conversation_store.save(conversation_id, state)
            logger.info(f"New conversation {conversation_id} initiated without message.") #
            return ChatResponse(
                conversation_id=conversation_id,
                current_agent=current_agent_name,
                messages=[],
                events=[],
                context=ctx.model_dump(),
                agents=_build_agents_list(),
                guardrails=[],
            )
    else:
        conversation_id = req.conversation_id  # type: ignore
        state = conversation_store.get(conversation_id)

    current_agent = _get_agent_by_name(state["current_agent"])
    state["input_items"].append({"content": req.message, "role": "user"})
    old_context = state["context"].model_dump().copy()
    guardrail_checks: List[GuardrailCheck] = []

    log_metric("user_query", {"conversation_id": conversation_id, "message": req.message})
    logger.info(f"Processing message for conversation {conversation_id} by {current_agent.name}: {req.message}") #

    try:
        result = await Runner.run(current_agent, state["input_items"], context=state["context"])
    except InputGuardrailTripwireTriggered as e:
        failed = e.guardrail_result.guardrail
        gr_output = e.guardrail_result.output.output_info
        gr_reasoning = getattr(gr_output, "reasoning", "")
        gr_input = req.message
        gr_timestamp = time.time() * 1000
        for g in current_agent.input_guardrails:
            guardrail_checks.append(GuardrailCheck(
                id=uuid4().hex,
                name=_get_guardrail_name(g),
                input=gr_input,
                reasoning=(gr_reasoning if g == failed else ""),
                passed=(g != failed),
                timestamp=gr_timestamp,
            ))
        refusal = "Sorry, I can only answer questions related to a telcomunnication company."
        state["input_items"].append({"role": "assistant", "content": refusal})

        log_metric("guardrail_triggered", {
            "conversation_id": conversation_id,
            "guardrail_name": _get_guardrail_name(e.guardrail_result.guardrail),
            "input": req.message,
            "reasoning": getattr(e.guardrail_result.output.output_info, "reasoning", "")
        })
        logger.warning(f"Guardrail '{_get_guardrail_name(failed)}' triggered for conversation {conversation_id}. Reason: {gr_reasoning}") #

        return ChatResponse(  
            conversation_id=conversation_id,
            current_agent=current_agent.name,
            messages=[MessageResponse(content=refusal, agent=current_agent.name)],
            events=[],
            context=state["context"].model_dump(),
            agents=_build_agents_list(),
            guardrails=guardrail_checks,
        )
    except Exception as e: # Catch all other exceptions
        logger.exception(f"An unexpected error occurred during agent run for conversation {conversation_id}: {e}") #
        error_message = "I'm sorry, an unexpected error occurred. Please try again later."
        state["input_items"].append({"role": "assistant", "content": error_message})
        return ChatResponse(
            conversation_id=conversation_id,
            current_agent=current_agent.name,
            messages=[MessageResponse(content=error_message, agent=current_agent.name)],
            events=[],
            context=state["context"].model_dump(),
            agents=_build_agents_list(),
            guardrails=guardrail_checks,
        )

    messages: List[MessageResponse] = []
    events: List[AgentEvent] = []

    for item in result.new_items:
        if isinstance(item, MessageOutputItem):
            text = ItemHelpers.text_message_output(item)
            messages.append(MessageResponse(content=text, agent=item.agent.name))
            events.append(AgentEvent(id=uuid4().hex, type="message", agent=item.agent.name, content=text))
            logger.info(f"Agent '{item.agent.name}' response: {text[:100]}...") #

        # Handle handoff output and agent switching
        elif isinstance(item, HandoffOutputItem):
            # Record the handoff event
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="handoff",
                    agent=item.source_agent.name,
                    content=f"{item.source_agent.name} -> {item.target_agent.name}",
                    metadata={"source_agent": item.source_agent.name, "target_agent": item.target_agent.name},
                )
            )

            log_metric("handoff", {
                "conversation_id": conversation_id,
                "source_agent": item.source_agent.name,
                "target_agent": item.target_agent.name
            })
            logger.info(f"Handoff: {item.source_agent.name} handed off to {item.target_agent.name} for conversation {conversation_id}") #

            # If there is an on_handoff callback defined for this handoff, show it as a tool call
            from_agent = item.source_agent
            to_agent = item.target_agent
            # Find the Handoff object on the source agent matching the target
            ho = next(
                (h for h in getattr(from_agent, "handoffs", [])
                 if isinstance(h, Handoff) and getattr(h, "agent_name", None) == to_agent.name),
                None,
            )
            if ho:
                fn = ho.on_invoke_handoff
                fv = fn.__code__.co_freevars
                cl = fn.__closure__ or []
                if "on_handoff" in fv:
                    idx = fv.index("on_handoff")
                    if idx < len(cl) and cl[idx].cell_contents:
                        cb = cl[idx].cell_contents
                        cb_name = getattr(cb, "__name__", repr(cb))
                        events.append(
                            AgentEvent(
                                id=uuid4().hex,
                                type="tool_call",
                                agent=to_agent.name,
                                content=cb_name,
                            )
                        )
                        
            current_agent = item.target_agent
        
        # Handle toolcallitem
        elif isinstance(item, ToolCallItem):
            tool_name = getattr(item.raw_item, "name", None)
            raw_args = getattr(item.raw_item, "arguments", None)
            tool_args: Any = raw_args
            if isinstance(raw_args, str):
                try:
                    import json
                    tool_args = json.loads(raw_args)
                except Exception as e:
                    logger.exception(f"Unexpected error parsing tool arguments: {e}") #
                    pass
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="tool_call",
                    agent=item.agent.name,
                    content=tool_name or "",
                    metadata={"tool_args": tool_args},
                )
            )

            # Log tool usage
            log_metric("tool_usage", {
                        "conversation_id": conversation_id,
                        "agent": item.agent.name,
                        "tool_name": tool_name,
                        "tool_args": tool_args
            })
            logger.info(f"Agent '{item.agent.name}' called tool '{tool_name}' with args: {tool_args}") #

        # Handle tool call output
        elif isinstance(item, ToolCallOutputItem):
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="tool_output",
                    agent=item.agent.name,
                    content=str(item.output),
                    metadata={"tool_result": item.output},
                )
            )
            logger.info(f"Agent '{item.agent.name}' received tool output: {str(item.output)[:100]}...") #

    new_context = state["context"].dict()
    changes = {k: new_context[k] for k in new_context if old_context.get(k) != new_context[k]}
    if changes:
        events.append(
            AgentEvent(
                id=uuid4().hex,
                type="context_update",
                agent=current_agent.name,
                content="",
                metadata={"changes": changes},
            )
        )
        logger.info(f"Context updated for conversation {conversation_id}: {changes}") #

    state["input_items"] = result.to_input_list()
    state["current_agent"] = current_agent.name
    conversation_store.save(conversation_id, state)
    logger.info(f"Conversation {conversation_id} state saved. Current agent: {current_agent.name}") #

    # Build guardrail results: mark failures (if any), and any others as passed
    final_guardrails: List[GuardrailCheck] = []
    for g in getattr(current_agent, "input_guardrails", []):
        name = _get_guardrail_name(g)
        failed = next((gc for gc in guardrail_checks if gc.name == name), None)
        if failed:
            final_guardrails.append(failed)
        else:
            final_guardrails.append(GuardrailCheck(
                id=uuid4().hex,
                name=name,
                input=req.message,
                reasoning="",
                passed=True,
                timestamp=time.time() * 1000,
            ))

    return ChatResponse(
        conversation_id=conversation_id,
        current_agent=current_agent.name,
        messages=messages,
        events=events,
        context=state["context"].dict(),
        agents=_build_agents_list(),
        guardrails=final_guardrails,
    )
