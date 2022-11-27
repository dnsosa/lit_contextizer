"""Object for loading XML and parsing marked up text."""

# -*- coding: utf-8 -*-

from xml.sax.handler import ContentHandler  # noqa: S406


class Xml2PseudoJson(ContentHandler):
    """Parser for XML marked up text."""

    def __init__(self):
        """Create XML to pseudo JSON object."""
        self.tags = []
        self.attributes = []
        self.chars = []
        self.event_dict = {}
        self.char_pos = 0
        self.locator = None

    def startElement(self, tag, attrs):  # noqa: N802
        """Keep track of attributes of parsed tags."""
        # Note: we're assuming the tree is only one level deep
        if tag != "root":
            self.tags.append(tag)
            self.attributes.append(dict(attrs))
        self.chars = []

    def setDocumentLocator(self, loc):  # noqa: N802
        """Create locator to keep track of data positions."""
        self.locator = loc

    def endElement(self, tag):  # noqa: N802
        """Update the extracted events when end tag is found."""
        if self.tags[:]:  # if self.tags isn't empty basically
            # d = {''.join(self.chars): self.tags[:]}
            attributes = self.attributes.pop()
            attributes["line_num"] = self.locator.getLineNumber()
            attributes["col_num"] = self.locator.getColumnNumber()
            attributes["pos"] = self.char_pos - 1

            # Update the events dict being maintained based on tag attributes
            event_text = ''.join(self.chars).lower()  # Get text of the event based on character list
            self.event_dict[event_text] = attributes  # NOTE: will keep overriding if there's repeats...

        # Refresh the character set and the tag attribute for the next new tag
        self.chars = []
        if len(self.tags) > 0:
            self.tags.pop()

    def characters(self, content):
        """Maintain characters in between tags and the character position."""
        self.chars.append(content)
        self.char_pos += len(content)
