"""Object for stripping tags from XML. NOTE: currently unused."""

# -*- coding: utf-8 -*-

from html.parser import HTMLParser
from io import StringIO


class MLStripper(HTMLParser):
    """Class for removing HTML-like tags from a string input."""

    def __init__(self):
        """Create new object for stripping tags."""
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        """Handle data."""
        self.text.write(d)

    def get_data(self):
        """Get data."""
        return self.text.getvalue()
