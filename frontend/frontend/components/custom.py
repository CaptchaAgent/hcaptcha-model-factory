# -*- coding: utf-8 -*-
# Time       : 2023/11/2 17:56
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import reflex as rx


class Spline(rx.Component):
    """Spline component."""

    library = "@splinetool/react-spline"
    tag = "Spline"
    # scene: rx.Var[str] = "https://prod.spline.design/Br2ec3WwuRGxEuij/scene.splinecode"
    scene: rx.Var[str] = "https://prod.spline.design/PuLetvrUjzCqpW7y/scene.splinecode"
    is_default = True

    lib_dependencies: list[str] = ["@splinetool/runtime"]


spline = Spline.create
