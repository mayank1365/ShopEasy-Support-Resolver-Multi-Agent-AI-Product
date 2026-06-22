"""External tools the agents can call.

Two tools, per the brief:
  1. order_lookup  — read-only; called by the Drafter (via Claude tool-use) when a
                     ticket references an order, so the reply can cite real order facts.
  2. send_reply    — high-impact write action; called ONLY after human approval.

Both are mocked with an in-memory store so the project runs end-to-end with no
external accounts. The Anthropic-facing JSON schema for order_lookup lives here too.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Mock "database" of orders. In a real system this would be an API/DB call.
# ---------------------------------------------------------------------------
_ORDERS: dict[str, dict] = {
    # Refundable: delivered recently, low value, not final sale -> clean approval.
    "ORD-1001": {
        "status": "delivered",
        "delivered_days_ago": 5,
        "amount": 49.99,
        "item": "Wireless Mouse",
        "payment_method": "Visa ****4242",
        "final_sale": False,
    },
    # Outside the 30-day window AND above the $200 threshold -> should be declined/escalated.
    "ORD-1002": {
        "status": "delivered",
        "delivered_days_ago": 45,
        "amount": 250.00,
        "item": "Mechanical Keyboard",
        "payment_method": "PayPal",
        "final_sale": False,
    },
    # Clearance / final sale -> not refundable even though in transit.
    "ORD-1003": {
        "status": "in_transit",
        "delivered_days_ago": None,
        "amount": 19.99,
        "item": "USB-C Cable (Clearance)",
        "payment_method": "Mastercard ****1111",
        "final_sale": True,
    },
    # Refundable: mid-value, comfortably inside the window.
    "ORD-1004": {
        "status": "delivered",
        "delivered_days_ago": 10,
        "amount": 129.00,
        "item": "Bluetooth Headphones",
        "payment_method": "Amex ****0005",
        "final_sale": False,
    },
    # IN WINDOW but ABOVE $200 -> eligible, but needs manager approval (tests the threshold rule).
    "ORD-1005": {
        "status": "delivered",
        "delivered_days_ago": 3,
        "amount": 399.00,
        "item": "4K Monitor",
        "payment_method": "Visa ****4242",
        "final_sale": False,
    },
    # Not delivered yet -> can't refund a delivered item; cancel/return flow instead.
    "ORD-1006": {
        "status": "in_transit",
        "delivered_days_ago": None,
        "amount": 75.00,
        "item": "Laptop Stand",
        "payment_method": "PayPal",
        "final_sale": False,
    },
    # Delivered long ago -> outside the 30-day window (tests window rule on a low-value item).
    "ORD-1007": {
        "status": "delivered",
        "delivered_days_ago": 60,
        "amount": 89.00,
        "item": "Office Chair",
        "payment_method": "Mastercard ****1111",
        "final_sale": False,
    },
}


def order_lookup(order_id: str) -> dict:
    """Look up an order by ID. Returns order facts or an error marker."""
    order = _ORDERS.get(order_id.strip().upper())
    if order is None:
        return {"found": False, "order_id": order_id}
    return {"found": True, "order_id": order_id.strip().upper(), **order}


# JSON schema advertised to Claude so it can call order_lookup during drafting.
ORDER_LOOKUP_TOOL = {
    "name": "order_lookup",
    "description": (
        "Look up the status, amount, item, delivery date, and refund-eligibility "
        "details of a customer order by its order ID (e.g. 'ORD-1001'). Call this "
        "whenever the customer references an order and the reply depends on its details."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order ID, e.g. 'ORD-1001'.",
            }
        },
        "required": ["order_id"],
    },
}


def send_reply(channel: str, message: str) -> str:
    """Mock the high-impact action of sending a reply to the customer.

    In production this would hit an email/Slack/helpdesk API. It is invoked only
    from the `send` node, which runs strictly after a human approves.
    """
    preview = message.replace("\n", " ")[:60]
    return f"Reply sent to customer via {channel}. (preview: '{preview}...')"
