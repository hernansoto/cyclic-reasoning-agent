from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import Settings
from llm import build_llm
from state import AgentState
from tools import TOOLS


SYSTEM_PROMPT = """
Eres un agente de operaciones con acceso a herramientas.

REGLAS:
1. Decide autónomamente si necesitas una herramienta. No afirmes datos de pedidos
   basándote en memoria general ni inventes valores.
2. Para preguntas sobre pedidos de clientes, usa las herramientas disponibles.
3. Lee cuidadosamente la salida de cada herramienta.
4. Si una salida indica `has_more=true`, la información está incompleta:
   vuelve a llamar a la herramienta con `next_page` antes de responder.
5. Si una herramienta devuelve un error recuperable y explica cómo corregirlo,
   realiza un segundo intento con parámetros válidos.
6. Si falta un dato imprescindible que no puedes deducir ni recuperar, pide una
   aclaración al usuario.
7. Cuando ya tengas todas las páginas necesarias, integra los resultados y
   responde de forma breve. Para importes, indica el total numérico claramente.
8. Nunca expongas razonamientos internos. Puedes mencionar qué datos consultaste,
   pero no describas cadenas privadas de pensamiento.
""".strip()


def build_agent(
    settings: Settings,
    checkpointer: Any,
):
    """
    Construye el StateGraph ReAct.

    Flujo:
        START -> model -> (tools_condition) -> tools -> model -> ... -> END
    """
    llm = build_llm(settings)
    llm_with_tools = llm.bind_tools(TOOLS)

    async def call_model(state: AgentState) -> dict[str, list]:
        response = await llm_with_tools.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                *state["messages"],
            ]
        )

        return {
            "messages": [response],
        }

    tool_node = ToolNode(
        TOOLS,
        handle_tool_errors=True,
    )

    builder = StateGraph(AgentState)

    builder.add_node("model", call_model)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "model")

    builder.add_conditional_edges(
        "model",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END,
        },
    )

    builder.add_edge(
        "tools",
        "model",
    )

    return builder.compile(
        checkpointer=checkpointer
    )
