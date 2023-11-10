# -*- coding: utf-8 -*-
# Time       : 2023/11/2 17:56
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from typing import Any, Dict, List

import reflex as rx
from reflex.components.component import Component
from reflex.vars import Var


class Spline(rx.Component):
    """Spline component."""

    library = "@splinetool/react-spline"
    tag = "Spline"
    # scene: rx.Var[str] = "https://prod.spline.design/Br2ec3WwuRGxEuij/scene.splinecode"
    scene: rx.Var[str] = "https://prod.spline.design/PuLetvrUjzCqpW7y/scene.splinecode"
    is_default = True

    lib_dependencies: list[str] = ["@splinetool/runtime"]


spline = Spline.create


class ReactFlowLib(Component):
    """A component that wraps a React flow lib."""

    library = "reactflow"

    def _get_custom_code(self) -> str | None:
        return """import 'reactflow/dist/style.css';
        """


class ReactFlow(ReactFlowLib):
    tag = "ReactFlow"

    nodes: Var[List[Dict[str, Any]]]
    edges: Var[List[Dict[str, Any]]]

    fit_view: Var[bool]
    nodes_draggable: Var[bool]
    nodes_connectable: Var[bool]
    nodes_focusable: Var[bool]

    def get_event_triggers(self) -> Dict[str, Any]:
        return {
            **super().get_event_triggers(),
            "on_edges_change": lambda e0: [e0],
            "on_connect": lambda e0: [e0],
        }


class Background(ReactFlowLib):
    tag = "Background"
    color: Var[str]
    gap: Var[str]

    size: Var[int]

    variant: Var[str]


class Controls(ReactFlowLib):
    tag = "Controls"


react_flow = ReactFlow.create
background = Background.create
controls = Controls.create
