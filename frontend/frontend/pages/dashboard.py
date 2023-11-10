"""The dashboard page."""
import random
from typing import List, Dict, Any

import reflex as rx

from frontend.components.custom import react_flow, background, controls
from frontend.templates import template

initial_nodes = [
    {"id": "1", "type": "input", "data": {"label": "150"}, "position": {"x": 250, "y": 25}},
    {"id": "2", "data": {"label": "25"}, "position": {"x": 100, "y": 125}},
    {"id": "3", "type": "output", "data": {"label": "5"}, "position": {"x": 250, "y": 250}},
]

initial_edges = [
    {"id": "e1-2", "source": "1", "target": "2", "label": "*", "animated": True},
    {"id": "e2-3", "source": "2", "target": "3", "label": "+", "animated": True},
]


class State(rx.State):
    """The app state."""

    nodes: List[Dict[str, Any]] = initial_nodes
    edges: List[Dict[str, Any]] = initial_edges

    def add_random_node(self):
        new_node_id = f"{len(self.nodes) + 1}"
        node_type = random.choice(["default"])
        # Label is random number
        label = new_node_id
        x = random.randint(0, 500)
        y = random.randint(0, 500)

        new_node = {
            "id": new_node_id,
            "type": node_type,
            "data": {"label": label},
            "position": {"x": x, "y": y},
            "draggable": True,
        }
        self.nodes.append(new_node)

    def clear_graph(self):
        self.nodes = []  # Clear the nodes list
        self.edges = []  # Clear the edges list

    def on_edges_change(self, new_edge):
        # Iterate over the existing edges
        for i, edge in enumerate(self.edges):
            # If we find an edge with the same ID as the new edge
            if edge["id"] == f"e{new_edge['source']}-{new_edge['target']}":
                # Delete the existing edge
                del self.edges[i]
                break

        # Add the new edge
        self.edges.append(
            {
                "id": f"e{new_edge['source']}-{new_edge['target']}",
                "source": new_edge["source"],
                "target": new_edge["target"],
                "label": random.choice(["+", "-", "*", "/"]),
                "animated": True,
            }
        )


@template(route="/dashboard", title="Dashboard")
def dashboard() -> rx.Component:
    """The dashboard page.

    Returns:
        The UI for the dashboard page.
    """
    # return rx.vstack(
    #     rx.heading("Dashboard", font_size="3em"),
    #     rx.text("Welcome to Reflex!"),
    #     rx.text("You can edit this page in ", rx.code("{your_app}/pages/dashboard.py")),
    # )
    return rx.center(
        rx.vstack(
            react_flow(
                background(),
                controls(),
                nodes_draggable=True,
                nodes_connectable=True,
                on_connect=lambda e0: State.on_edges_change(e0),
                nodes=State.nodes,
                edges=State.edges,
                fit_view=True,
            ),
            rx.hstack(
                rx.button("Clear graph", on_click=State.clear_graph, width="100%"),
                rx.button("Add node", on_click=State.add_random_node, width="100%"),
                width="100%",
            ),
            height="30em",
            width="100%",
        )
    )
