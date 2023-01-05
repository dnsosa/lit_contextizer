"""
Class for storing extractable events from text, namely context mentions and relations.

Note that Extractables will only exist in the context of the paper that "holds" it.
"""

# -*- coding: utf-8 -*-

from lit_contextizer.data_models.Sentence import Sentence


class Extractable:
    """An abstract representation of an extractable event from a larger textual context."""

    def __init__(self, text: str, paper_doi: str, paper_pmcid: str, paper_pmid: str, start_idx: int, end_idx: int,
                 sent_idx: int, sentence: Sentence):
        """
        Construct an extractable.

        :param text: string representation of the extractable
        :param paper_doi: associated paper's DOI
        :param paper_pmcid: associated paper's PMC ID
        :param start_idx: start character index of extractable in the full text
        :param end_idx: end character index of extractable in the full text
        :param sent_idx: sentence index of the extractable in the full text
        :param sentence: sentence object in which extractable is contained
        """
        self.text = text
        self.paper_doi = paper_doi
        self.paper_pmcid = paper_pmcid
        self.paper_pmid = paper_pmid
        self.start_idx = start_idx  # character index
        self.end_idx = end_idx
        self.sent_idx = sent_idx
        self.sentence = sentence

    def get_text(self):
        """Get text of extractable."""
        return self.text

    def get_paper_doi(self):
        """Get DOI of paper containing extractable."""
        return self.paper_doi

    def get_paper_pmcid(self):
        """Get PMC ID of paper containing extractable."""
        return self.paper_pmcid

    def get_paper_pmid(self):
        """Get PMID of paper containing extractable."""
        return self.paper_pmid

    def get_start_idx(self):
        """Get start index."""
        return self.start_idx

    def get_end_idx(self):
        """Get end index."""
        return self.end_idx

    def get_sent_idx(self):
        """Get sentence index of extractable in context of full text."""
        return self.sent_idx

    def get_sentence(self):
        """Get sentence object containing extracted event."""
        return self.sentence


class Context(Extractable):
    """A representation of a context mention."""

    def __init__(self, uuid, ctx_type, ctx_uid, **kwds):
        """
        Construct a context object.

        :param uuid: context uuids provided by BAI
        :param ctx_type: context type as annotated in the context tags
        :param ctx_uid: a unique identifier which will be used to map Contexts to their Sentences in a Paper
        """
        super().__init__(**kwds)
        self.uuid = uuid
        self.ctx_type = ctx_type
        self.ctx_uid = ctx_uid

    def get_uuid(self):
        """Get Context BAI UUID."""
        return self.uuid

    def get_ctx_type(self):
        """Get context type."""
        return self.ctx_type

    def get_ctx_uid(self):
        """Get context paper UID."""
        return self.ctx_uid


class Relation(Extractable):
    """A representation of an extracted relation."""

    def __init__(self, main_verb: str, entity1: str, entity2: str, **kwds):
        """
        Construct a relation object.

        :param main_verb: main verb of relation (e.g. "upregulates")
        :param entity1: entity 1 of the relationship
        :param entity2: entity 2 of the relationship
        """
        super().__init__(**kwds)
        self.main_verb = main_verb
        self.entity1 = entity1
        self.entity2 = entity2

    def get_main_verb(self):
        """Get the main verb from the relation."""
        return self.main_verb

    def get_entity1(self):
        """Get entity 1."""
        return self.entity1

    def get_entity2(self):
        """Gett enity 2."""
        return self.entity2
