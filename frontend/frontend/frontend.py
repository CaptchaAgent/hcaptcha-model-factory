"""Welcome to Reflex!."""

from frontend import styles

# Import all the pages.
from frontend.pages import *

import reflex as rx

# Create the app and compile it.
app = rx.App(style=styles.base_style)
app.compile()
