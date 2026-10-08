from __future__ import annotations

import json
import math
from typing import TypedDict

from langchain_core.tools import tool


class Order(TypedDict):
    order_id: str
    date: str
    product: str
    amount: int


ORDERS_DB: dict[int, list[Order]] = {
    101: [
        {
            "order_id": "ORD-101-001",
            "date": "2026-08-03",
            "product": "Teclado mecánico",
            "amount": 3100,
        }
    ],
    102: [
        {
            "order_id": "ORD-102-001",
            "date": "2026-08-10",
            "product": "Monitor 24 pulgadas",
            "amount": 4200,
        },
        {
            "order_id": "ORD-102-002",
            "date": "2026-08-22",
            "product": "Dock USB-C",
            "amount": 5300,
        },
        {
            "order_id": "ORD-102-003",
            "date": "2026-09-05",
            "product": "Webcam 4K",
            "amount": 5000,
        },
    ],
    103: [
        {
            "order_id": "ORD-103-001",
            "date": "2026-09-12",
            "product": "Mouse ergonómico",
            "amount": 2500,
        },
        {
            "order_id": "ORD-103-002",
            "date": "2026-09-28",
            "product": "Auriculares USB",
            "amount": 4600,
        },
    ],
}

PAGE_SIZE = 2


@tool
async def buscar_pedidos_cliente(
    cliente_id: int,
    pagina: int = 1,
) -> str:
    """
    Consulta la base simulada de pedidos de un cliente.

    Debes usar esta herramienta cuando el usuario pregunte por pedidos,
    cantidades de pedidos, importes, productos, fechas, totales o el último
    pedido de un cliente identificado por su número.

    IMPORTANTE: la herramienta es PAGINADA y devuelve como máximo dos pedidos
    por llamada. Si la respuesta contiene `"has_more": true`, la información es
    INCOMPLETA y NO debes calcular ni responder totales definitivos todavía.
    Debes llamar nuevamente ESTA MISMA herramienta con el mismo `cliente_id` y
    con la `pagina` indicada en `"next_page"`. Repite hasta recibir
    `"has_more": false`.

    Si recibes un error recuperable (por ejemplo una página inválida), lee
    `"valid_pages"` y vuelve a intentar con un valor permitido. No inventes
    pedidos, montos ni páginas.

    Args:
        cliente_id: Identificador numérico exacto del cliente.
        pagina: Página a consultar. Comienza en 1.

    Returns:
        Un JSON serializado con los pedidos de esa página, cantidad parcial,
        subtotal parcial, indicadores de paginación y datos para continuar.
    """
    orders = ORDERS_DB.get(cliente_id)

    if orders is None:
        return json.dumps(
            {
                "ok": False,
                "error": "cliente_no_encontrado",
                "cliente_id": cliente_id,
                "message": (
                    "No existe ese cliente. Verifica el identificador "
                    "o pide una aclaración al usuario."
                ),
            },
            ensure_ascii=False,
        )

    total_pages = max(1, math.ceil(len(orders) / PAGE_SIZE))

    if pagina < 1 or pagina > total_pages:
        return json.dumps(
            {
                "ok": False,
                "error": "pagina_invalida",
                "cliente_id": cliente_id,
                "pagina_recibida": pagina,
                "valid_pages": list(range(1, total_pages + 1)),
                "message": (
                    "La página solicitada no existe. Reintenta usando "
                    "una página incluida en valid_pages."
                ),
            },
            ensure_ascii=False,
        )

    start = (pagina - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    page_orders = orders[start:end]

    has_more = pagina < total_pages

    payload = {
        "ok": True,
        "cliente_id": cliente_id,
        "page": pagina,
        "total_pages": total_pages,
        "orders_on_page": len(page_orders),
        "subtotal_page": sum(
            order["amount"] for order in page_orders
        ),
        "orders": page_orders,
        "has_more": has_more,
        "next_page": pagina + 1 if has_more else None,
        "instruction": (
            "Información incompleta: consulta next_page antes de "
            "responder totales definitivos."
            if has_more
            else "Última página: ya puedes razonar con todas las páginas."
        ),
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
    )


TOOLS = [buscar_pedidos_cliente]
