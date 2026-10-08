# Cyclic Reasoning Agent — Pre-entrega 5

Agente ReAct construido con Python 3.12, LangGraph, herramientas LangChain y
memoria persistente SQLite.

El objetivo es demostrar:

- autonomía para decidir cuándo usar herramientas;
- razonamiento cíclico `modelo → herramienta → modelo`;
- segundo intento cuando la información de una herramienta es incompleta;
- persistencia conversacional mediante `thread_id`;
- ejecución asíncrona;
- límite explícito de recursión;
- traza observable en JSON.

## Arquitectura

```text
                 ┌───────────────────────────────────┐
                 │            MessagesState          │
                 │  historial + ToolMessages         │
                 └────────────────┬──────────────────┘
                                  │
                                  ▼
START ──────────────────────────> model
                                  │
                           tools_condition
                           ┌──────┴──────┐
                           │             │
                    tool_calls       sin tool call
                           │             │
                           ▼             ▼
                         tools          END
                           │
                           └──────────> model
                                        │
                                        └── ciclo
```

La decisión de utilizar herramientas no se realiza mediante `if/else` sobre el
prompt del usuario. El LLM recibe las herramientas con `bind_tools()` y decide
autónomamente si genera un `tool_call`.

## Estado

El estado hereda explícitamente de:

```python
class AgentState(MessagesState):
    pass
```

`MessagesState` utiliza el reducer de mensajes de LangGraph para acumular la
conversación y los resultados de herramientas.

## Herramienta propia

El proyecto implementa:

```python
@tool
async def buscar_pedidos_cliente(
    cliente_id: int,
    pagina: int = 1,
) -> str:
    ...
```

La herramienta simula una consulta a base de datos.

Su docstring explica:

- cuándo debe utilizarla el modelo;
- que la respuesta está paginada;
- cómo detectar información incompleta;
- cómo continuar con `next_page`;
- cómo recuperarse de una página inválida.

## Multi-paso obligatorio

El cliente `102` tiene tres pedidos.

La herramienta devuelve un máximo de dos pedidos por llamada.

La pregunta de demostración es:

```text
¿Cuántos pedidos tuvo el cliente 102, cuál fue el importe total
de todos sus pedidos y qué producto compró en el último?
```

La primera llamada devuelve:

```json
{
  "page": 1,
  "has_more": true,
  "next_page": 2
}
```

Por lo tanto, el agente todavía no posee información suficiente para calcular
el total definitivo.

Debe volver al ciclo y realizar:

```text
buscar_pedidos_cliente(cliente_id=102, pagina=2)
```

Solo cuando recibe:

```json
{
  "page": 2,
  "has_more": false
}
```

puede concluir.

El resultado correcto con los datos simulados es:

```text
3 pedidos
total: $14.500
último producto: Webcam 4K
```

Así se demuestra una ejecución ReAct con al menos dos tool calls.

## StateGraph

El grafo se construye explícitamente:

```python
builder = StateGraph(AgentState)

builder.add_node("model", call_model)
builder.add_node("tools", ToolNode(TOOLS))

builder.add_edge(START, "model")

builder.add_conditional_edges(
    "model",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END,
    },
)

builder.add_edge("tools", "model")
```

El modelo se vincula a las herramientas mediante:

```python
llm_with_tools = llm.bind_tools(TOOLS)
```

## Persistencia SQLite

LangGraph proporciona dos implementaciones SQLite relacionadas:

- `SqliteSaver`: API sincrónica;
- `AsyncSqliteSaver`: API asíncrona.

Este proyecto incluye ambas en `persistence.py`.

Como el checkpoint exige `asyncio`, el runtime principal usa:

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
```

y:

```python
async with AsyncSqliteSaver.from_conn_string(
    "./agent_checkpoints.sqlite"
) as checkpointer:
    ...
```

Esto mantiene SQLite como mecanismo de persistencia, pero utiliza su interfaz
correcta para `ainvoke` y `astream`.

El archivo se genera localmente:

```text
agent_checkpoints.sqlite
```

y está excluido de Git.

## Memoria mediante `thread_id`

Cada ejecución pasa:

```python
config = {
    "configurable": {
        "thread_id": "cliente-102-demo"
    },
    "recursion_limit": 10,
}
```

El checkpointer relaciona los checkpoints con ese `thread_id`.

Después del primer turno:

```text
¿Cuántos pedidos tuvo el cliente 102...?
```

el demo realiza, con el MISMO thread:

```text
¿Y en qué fecha fue ese último pedido?
```

El historial anterior permanece disponible.

Incluso se puede cerrar el proceso y volver a ejecutar una consulta usando el
mismo archivo SQLite y el mismo `thread_id`.

## Recursion limit

Cada invocación incluye:

```python
"recursion_limit": 10
```

Esto evita ciclos infinitos y costos inesperados.

Puede modificarse en `.env`:

```env
RECURSION_LIMIT=10
```

La configuración Pydantic impone límites razonables.

## LLM

La consigna requiere OpenAI o Anthropic.

El proyecto soporta ambos:

```env
LLM_PROVIDER=openai
```

o:

```env
LLM_PROVIDER=anthropic
```

La arquitectura del grafo es idéntica para ambos.

## Estructura

```text
cyclic-reasoning-agent/
├── traces/
│   └── example_trace.json
├── .env.example
├── .gitignore
├── agent.py
├── chat.py
├── config.py
├── llm.py
├── main.py
├── persistence.py
├── state.py
├── tools.py
├── trace_utils.py
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.12+
- API key de OpenAI o Anthropic

## 1. Crear entorno virtual

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

## 2. Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Configurar `.env`

Copiar:

### Windows

```powershell
Copy-Item .env.example .env
```

### Linux / macOS

```bash
cp .env.example .env
```

### OpenAI

```env
LLM_PROVIDER=openai

OPENAI_API_KEY=tu_api_key
ANTHROPIC_API_KEY=

OPENAI_MODEL=gpt-5-mini
ANTHROPIC_MODEL=claude-sonnet-4-6

CHECKPOINT_DB=./agent_checkpoints.sqlite
RECURSION_LIMIT=10
```

### Anthropic

```env
LLM_PROVIDER=anthropic

OPENAI_API_KEY=
ANTHROPIC_API_KEY=tu_api_key

OPENAI_MODEL=gpt-5-mini
ANTHROPIC_MODEL=claude-sonnet-4-6

CHECKPOINT_DB=./agent_checkpoints.sqlite
RECURSION_LIMIT=10
```

Nunca subir `.env`.

## 4. Ejecutar demo completa

```bash
python main.py
```

El demo realiza:

1. una pregunta que obliga a consultar dos páginas;
2. una segunda pregunta con el mismo `thread_id`;
3. guarda la traza observable en JSON.

También puede indicarse otro thread:

```bash
python main.py --thread-id sesion-ventas-001
```

## 5. Conversar con un thread persistente

Primera ejecución:

```bash
python chat.py --thread-id cliente-102 "¿Cuántos pedidos tuvo el cliente 102?"
```

Después, incluso en otra ejecución del proceso:

```bash
python chat.py --thread-id cliente-102 "¿Y cuál fue el último?"
```

Al utilizar el mismo:

```text
thread_id = cliente-102
```

LangGraph recupera el estado almacenado en SQLite.

Con un thread diferente:

```bash
python chat.py --thread-id otra-sesion "¿Y cuál fue el último?"
```

no comparte el historial de `cliente-102`.

## Resiliencia frente a información incompleta

La herramienta devuelve explícitamente:

```json
{
  "has_more": true,
  "next_page": 2
}
```

cuando todavía quedan resultados.

El modelo recibe instrucciones para volver a llamar la herramienta antes de
responder.

Esto implementa el ciclo:

```text
model
  ↓
tool página 1
  ↓
model detecta información incompleta
  ↓
tool página 2
  ↓
model
  ↓
respuesta
```

## Resiliencia frente a error recuperable

Si el modelo solicita una página inválida, la herramienta devuelve:

```json
{
  "ok": false,
  "error": "pagina_invalida",
  "valid_pages": [1, 2],
  "message": "..."
}
```

El docstring instruye al modelo a reintentar con un valor incluido en
`valid_pages`.

Si el cliente no existe, el resultado indica que debe verificar el ID o pedir
aclaración al usuario.

## Traza ReAct

El repo incluye:

```text
traces/example_trace.json
```

La traza contiene únicamente eventos observables:

- mensajes del usuario;
- tool calls;
- argumentos de las herramientas;
- resultados de herramientas;
- respuestas finales.

No contiene chain-of-thought ni razonamiento privado.

Al ejecutar:

```bash
python main.py
```

también se genera:

```text
traces/latest_trace.json
```

Este último se ignora en Git para que el ejemplo versionado sea reproducible.

## Ejemplo resumido de traza

```text
Usuario
  ↓
buscar_pedidos_cliente(102, pagina=1)
  ↓
has_more=true / next_page=2
  ↓
buscar_pedidos_cliente(102, pagina=2)
  ↓
has_more=false
  ↓
Respuesta final
```

## Código asíncrono

Los puntos principales utilizan `async`:

```python
async def buscar_pedidos_cliente(...)

async def call_model(...)

await llm_with_tools.ainvoke(...)

async for event in graph.astream(...)
```

La persistencia usa `AsyncSqliteSaver`.

## Tipado

El proyecto utiliza:

- `Type Hints`;
- `TypedDict`;
- modelos Pydantic para configuración;
- tipos explícitos en herramientas y helpers.

## Seguridad

`.gitignore` excluye:

```text
.env
.venv/
*.sqlite
*.sqlite3
*.db
```

Antes del push:

```bash
git check-ignore .env
```

debe devolver:

```text
.env
```

## Criterios de aceptación

| Criterio | Implementación |
| --- | --- |
| Python 3.12 | Sí |
| StateGraph | Sí |
| Hereda de MessagesState | `AgentState(MessagesState)` |
| Nodo de modelo | `model` |
| Nodo de tools | `ToolNode` |
| Arista condicional | `tools_condition` |
| `@tool` personalizada | `buscar_pedidos_cliente` |
| Docstring descriptivo | Sí |
| `llm.bind_tools()` | Sí |
| Autonomía | LLM decide tool calls |
| Sin rutas por prompt | Sí |
| Ciclo de retorno | `tools → model` |
| Tool llamada ≥2 veces | Demo paginada |
| Error recuperable | página inválida |
| Pedir aclaración | cliente inexistente |
| SQLite persistente | Sí |
| `SqliteSaver` incluido | Sí |
| Async SQLite | `AsyncSqliteSaver` |
| `thread_id` | Sí |
| Memoria multi-turn | Sí |
| `recursion_limit` | 10 |
| Asyncio | Sí |
| Type hints | Sí |
| Traza JSON | Sí |
| API keys fuera de Git | Sí |

## Flujo de validación recomendado

```bash
python main.py
```

Luego:

```bash
python chat.py --thread-id prueba-1 "¿Cuántos pedidos tuvo el cliente 102?"
```

y:

```bash
python chat.py --thread-id prueba-1 "¿Qué compró en el último?"
```

Finalmente revisar:

```text
traces/example_trace.json
```

y confirmar que `.env` no está versionado.
