import argparse
import json
import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse, propagate_attributes
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

# Force .env values to win over any stale shell exports.
load_dotenv(override=True)

# Shared LLM instance. Configure in .env for OpenAI/OpenRouter compatibility.
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))


def _valid_key(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if not value or value.lower().startswith("your_"):
        return None
    return value


API_KEY = _valid_key(os.getenv("OPENAI_API_KEY")) or _valid_key(
    os.getenv("OPENROUTER_API_KEY")
)

if not API_KEY:
    raise RuntimeError(
        "Missing API key. Set OPENAI_API_KEY (or OPENROUTER_API_KEY) in .env."
    )


def _setup_langfuse():
    """Setup Langfuse tracing if credentials are provided."""
    langfuse_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not langfuse_key or not langfuse_secret:
        print("[INFO] Langfuse not configured. Skipping observability tracing.")
        return None

    try:
        client = Langfuse(
            public_key=langfuse_key,
            secret_key=langfuse_secret,
            host=langfuse_host,
        )
        print("[INFO] Langfuse tracing enabled.")
        return client
    except Exception as e:
        print(f"[WARNING] Failed to initialize Langfuse: {e}")
        return None


_langfuse_client = _setup_langfuse()


def _run_tool_with_trace(tool_name: str, payload: dict, fn):
    """Run a tool function with optional Langfuse tool observation."""
    if not _langfuse_client:
        return fn()

    span = _langfuse_client.start_observation(
        name=tool_name,
        as_type="tool",
        input=payload,
    )
    try:
        result = fn()
        span.update(output=result)
        return result
    except Exception as e:
        span.update(level="ERROR", status_message=str(e))
        raise
    finally:
        span.end()


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return _run_tool_with_trace(
        "add",
        {"a": a, "b": b},
        lambda: a + b,
    )


@tool
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first number."""
    return _run_tool_with_trace(
        "subtract",
        {"a": a, "b": b},
        lambda: a - b,
    )


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return _run_tool_with_trace(
        "multiply",
        {"a": a, "b": b},
        lambda: a * b,
    )


@tool
def divide(a: float, b: float) -> str:
    """Divide the first number by the second number."""
    def _do_divide() -> str:
        if b == 0:
            return "Error: division by zero is not allowed."
        return str(a / b)

    return _run_tool_with_trace(
        "divide",
        {"a": a, "b": b},
        _do_divide,
    )


TOOLS = [add, subtract, multiply, divide]

SYSTEM_PROMPT = """
You are a reasoning math agent built with LangGraph.
Always follow the bodmass order of operations when doing calculations.
Use the available math tools for arithmetic instead of doing calculations mentally.
Work step by step when needed, and give the final answer clearly.
If a tool returns an error, explain it to the user directly.

- Always follow order of operations (BODMAS)
- Break expression into atomic operations
- Use tools for EACH step
- Do not combine steps mentally

""".strip()

_llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    openai_api_key=API_KEY,
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
)

_tool_llm = _llm.bind_tools(TOOLS)
_required_tool_llm = _llm.bind_tools(TOOLS, tool_choice="required")


def _looks_like_math_query(text: str) -> bool:
    return bool(
        re.search(
            r"\d|[+\-*/()]|\b(add|plus|sum|subtract|minus|difference|multiply|times|product|divide|divided)\b",
            text,
            re.IGNORECASE,
        )
    )


def _message_role(message) -> str:
    if isinstance(message, HumanMessage):
        return "human"
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, ToolMessage):
        return "tool"
    return message.__class__.__name__.lower()


def _message_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def _print_llm_step(input_messages, ai_message) -> None:
    print("\n[LLM INPUT]")
    for idx, msg in enumerate(input_messages, start=1):
        print(f"{idx}. [{_message_role(msg)}] {_message_text(msg)}")

    print("[LLM OUTPUT]")
    print(f"[{_message_role(ai_message)}] {_message_text(ai_message)}")
    if getattr(ai_message, "tool_calls", None):
        print(f"tool_calls: {ai_message.tool_calls}")
    print()


def call_model(state: MessagesState) -> dict:
    """Reason over the conversation and decide whether a tool call is needed."""
    last_message = state["messages"][-1]
    llm = _tool_llm
    if isinstance(last_message, HumanMessage) and _looks_like_math_query(last_message.content):
        llm = _required_tool_llm
    if isinstance(last_message, ToolMessage):
        llm = _tool_llm
    input_messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    
    span = None
    if _langfuse_client:
        span = _langfuse_client.start_observation(
            name="assistant_reasoning",
            as_type="generation",
            input=_message_text(last_message),
            model=MODEL_NAME,
            model_parameters={"temperature": TEMPERATURE},
        )
    
    try:
        ai_message = llm.invoke(input_messages)
        _print_llm_step(input_messages, ai_message)
        if span:
            span.update(
                output=_message_text(ai_message),
                metadata={"tool_calls": getattr(ai_message, "tool_calls", None)},
            )
            span.end()
    except Exception as e:
        if span:
            span.update(level="ERROR", status_message=str(e))
            span.end()
        raise
    
    return {"messages": [ai_message]}


def route_after_model(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def create_agent():
    graph = StateGraph(MessagesState)
    graph.add_node("assistant", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", route_after_model, {"tools": "tools", END: END})
    graph.add_edge("tools", "assistant")
    return graph.compile()


def export_graph_mermaid(app, output_path: str) -> Path:
    graph_path = Path(output_path)
    graph_path.write_text(app.get_graph().draw_mermaid())
    return graph_path


def export_graph_png(app, output_path: str) -> Path:
    graph_path = Path(output_path)
    graph_path.write_bytes(app.get_graph().draw_mermaid_png())
    return graph_path


def parse_args():
    parser = argparse.ArgumentParser(description="LangGraph math agent")
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Print the compiled graph in Mermaid format.",
    )
    parser.add_argument(
        "--graph-file",
        help="Write the compiled graph Mermaid text to a file.",
    )
    parser.add_argument(
        "--graph-png",
        help="Write the compiled graph as a PNG image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = create_agent()

    if args.graph:
        print(app.get_graph().draw_mermaid())
        return

    if args.graph_file:
        graph_path = export_graph_mermaid(app, args.graph_file)
        print(f"Wrote Mermaid graph to {graph_path}")
        return

    if args.graph_png:
        graph_path = export_graph_png(app, args.graph_png)
        print(f"Wrote graph PNG to {graph_path}")
        return

    state = {"messages": []}
    langfuse_session_id = os.getenv("LANGFUSE_SESSION_ID") or f"math-agent-{uuid.uuid4().hex[:12]}"

    print("LangGraph math agent. Type 'exit' to quit.")
    if _langfuse_client:
        print(f"[INFO] Langfuse session_id: {langfuse_session_id}")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            print("Agent: Bye!")
            if _langfuse_client:
                _langfuse_client.flush()
            break

        if _langfuse_client:
            with _langfuse_client.start_as_current_observation(
                name="agent_turn",
                as_type="chain",
                input=user_text,
            ) as turn_span:
                with propagate_attributes(
                    session_id=langfuse_session_id,
                    trace_name="langgraph_math_agent",
                ):
                    state = app.invoke(
                        {"messages": state["messages"] + [HumanMessage(content=user_text)]}
                    )

                assistant_output = _message_text(state["messages"][-1])
                turn_span.update(output=assistant_output)
            _langfuse_client.flush()
        else:
            state = app.invoke(
                {"messages": state["messages"] + [HumanMessage(content=user_text)]}
            )

        print(f"Agent: {state['messages'][-1].content}")


if __name__ == "__main__":
    main()
