"""Class for storing data related to sentences and performing basic linguistic operations."""

# -*- coding: utf-8 -*-

import en_core_web_sm

nlp = en_core_web_sm.load()


class Sentence:
    """Object for storing text and enabling basic linguistic computation."""

    # Note, creating this might be really slow! Should only extract features when I need to.
    def __init__(self, text: str):
        """
        Create a sentence object.

        :param text: sentence text
        """
        self.text = text
        self.doc = None
        self.pov = None
        self.tense = None
        self.edge_list = None
        self.root = None

    def extract_pov(self) -> set():
        """
        Extract all point of views (persons) (POVs) present in sentence by analyzing pronouns.

        :return: set of all POVs present in sentence
        """
        if self.doc is None:
            self.doc = nlp(self.text)
        pov_set = set()
        for token in self.doc:
            if token.pos_ == "PRON" and token.has_morph():
                if len(token.morph.get("Person")) > 0:  # Sometimes there's no "person" morphology info
                    pov = token.morph.get("Person")[0]
                    pov_set.add(pov)

        return pov_set

    def extract_tense(self) -> str:
        """
        Attempt to extract the tense of the string by looking for the presence of past tense verbs.

        :return: indication of sentence in past or present tense
        """
        # only looking for if ANY VBN or VBD exists in the sentence--a heuristic.
        # TODO: Decide rule if multiple tenses exist
        if self.doc is None:
            self.doc = nlp(self.text)

        for token in self.doc:
            if token.tag_ in ["VBN", "VBD"]:
                return "past"
        return "present"

    def extract_edges(self):
        """Extract edge list based on Spacy's dependency path of the sentence text."""
        if self.doc is None:
            self.doc = nlp(self.text)

        edge_list = []
        for token in self.doc:
            for child in token.children:
                edge_list.append((f"{token.lower_}", f"{child.lower_}"))
        return edge_list

    def extract_root(self):
        """Find root of dependency path."""
        if self.doc is None:
            self.doc = nlp(self.text)

        for token in self.doc:
            if token.head == token:
                return token  # Note: outputs a Spacy Token (need to convert to string later)
        return None

    def get_text(self):
        """Get sentence text."""
        return self.text

    def get_pov(self):
        """Get sentence point of view."""
        if self.pov is None:
            self.pov = self.extract_pov()
        return self.pov

    def get_tense(self):
        """Get sentence tense."""
        if self.tense is None:
            self.tense = self.extract_tense()
        return self.tense

    def get_edge_list(self):
        """Get dependency path edge list."""
        if self.edge_list is None:
            self.edge_list = self.extract_edges()
        return self.edge_list

    def get_root(self):
        """Get root of sentence dependency path."""
        if self.root is None:
            self.root = self.extract_root()
        return self.root

    # Make sure to make a feature for each the context and the relationship sentence
    def is_first_person(self):
        """Return whether or not sentence is in first person."""
        # check this is right
        return "1" in self.get_pov()

    def is_past_tense(self):
        """Return whether or not sentence is in past tense."""
        # tag the parts of speech of sentence
        # check this
        return "past" in self.get_tense()

    def is_present_tense(self):
        """Return whether or not sentence is in present tense."""
        # tag the parts of speech of sentence
        # check this
        return "present" in self.get_tense()
