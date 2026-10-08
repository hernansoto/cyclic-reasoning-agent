from __future__ import annotations

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """
    Estado del agente.

    Hereda explícitamente de MessagesState, cuyo reducer `add_messages`
    acumula la conversación y los ToolMessage sin reemplazar el historial.
    """

    pass
