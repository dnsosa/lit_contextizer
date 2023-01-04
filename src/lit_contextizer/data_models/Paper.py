"""Class for storing data related to scientific publications."""

# -*- coding: utf-8 -*-

from os import path

import bioc

from lit_contextizer.data_models.Extractable import Context, Extractable, Relation
from lit_contextizer.data_models.Sentence import Sentence
from lit_contextizer.data_models.constants import BACKGROUND_SYNS_PATH, DISCCONC_SYNS_PATH, METHODS_SYNS_PATH, \
    RESULTS_SYNS_PATH

import matplotlib.pyplot as plt

import networkx as nx

import numpy as np

import pandas as pd

from rapidfuzz import fuzz, process

local_biocxml_dir = "/Users/dnsosa/Desktop/AltmanLab/bai/biotext/full_texts/PM_files"

background_syns = set(pd.read_csv(BACKGROUND_SYNS_PATH).synonyms)
methods_syns = set(pd.read_csv(METHODS_SYNS_PATH).synonyms)
results_syns = set(pd.read_csv(RESULTS_SYNS_PATH).synonyms)
discconc_syns = set(pd.read_csv(DISCCONC_SYNS_PATH).synonyms)


class Paper:
    """Class for representing a scientific paper."""

    def __init__(self, title, abstract, full_text, pmid=None, pmcid=None, doi=None, date=None, journal=None):
        """Construct paper object."""
        self.title = title
        self.abstract = abstract
        self.full_text = full_text
        self.full_text_in_sections = []
        self.pmcid = pmcid
        self.pmid = pmid  # NOTE: PMID is used for section mapper
        self.doi = doi
        self.date = date
        self.journal = journal

        if self.full_text is not None:
            self.full_text_sents = [s + "." for s in self.full_text.split(". ") if s]
        elif self.abstract is not None:
            self.full_text_sents = [s + "." for s in self.abstract.split(". ") if s]
        else:
            self.full_text_sents = None

        self.context_list = []
        self.relations = []

        self.sec_mapper = {}
        self.load_sec_mapper()

    def load_sec_mapper(self):
        """Load BioCXML file into a dictionary representation to enable quick querying of section info by sentence."""
        biocxml_file = f"{local_biocxml_dir}/{self.pmcid}.biocxml"

        if path.exists(biocxml_file):
            with open(biocxml_file, 'rb') as f:
                # Parser should have a single paper, just get that one paper.
                # parser = bioc.BioCXMLDocumentReader(f)
                # document = parser.__next__()
                parser = bioc.biocxml.load(f)
                document = parser.documents[0]
                for sec_idx, sec in enumerate(document.passages):
                    self.full_text_in_sections.append(sec.text)
                    # Get the name of the section
                    if sec.infons['section'] != 'article':  # 'article' means body of text. Want to catch title/abs
                        sec_type = sec.infons['section']
                    else:  # get the name of the section e.g. methods, results
                        sec_type = sec.infons['subsection']
                    sent_list = [s + "." for s in sec.text.split(". ") if s]
                    for sent in sent_list:
                        self.sec_mapper[sent] = (sec_idx, sec_type)
        else:
            print(f"BiocXML file: {biocxml_file} couldn't be opened. No section mapper loaded for Paper {self.pmcid}.")

        return None

    def set_context_list(self, context_list: list):
        """Set context list variable to the list of context objects that has been preprocessed."""
        self.context_list = context_list

    def add_relation(self, relation: Relation):
        """Add a relation to the relation_list."""
        self.relations.append(relation)

    def find_sentence_idx(self, in_sentence: Sentence, threshold: float = 80.0) -> int:
        """
        Given a sentence, find the index of that sentence in the full body text via fuzzy matching.

        :param in_sentence: Sentence being queried for
        :param threshold: threshold (0-100) below which the match quality is too poor
        :return: sentence index, match score, and the top sentence found
        """
        if self.full_text_sents is None:
            return None, None, None

        in_sent_text = in_sentence.get_text()
        found_sent, score, idx = process.extractOne(in_sent_text, self.full_text_sents, scorer=fuzz.ratio)

        # No really good sentence was found for some reason (e.g.
        if score < threshold:  # arbitrary
            return None, score, found_sent
        else:
            return idx, score, found_sent

    def get_section_info(self, in_sentence: Sentence, threshold: int = 80.0):
        """
        Given a sentence, determine which section it's from.

        :param in_sentence: Sentence being queried for
        :param threshold: threshold (0-100) below which the match quality is too poor
        :return: section index, section type, score of fuzzy match, and sentence found in section mapper
        """
        if (self.full_text_sents is None) or (len(self.sec_mapper.keys()) == 0):
            return None, None, None, None

        # Get the text of the sentence object
        in_sent_text = in_sentence.get_text()
        # Compare against all the section information that was extracted--find the closest match that's above threshold
        found_sent, score, idx = process.extractOne(in_sent_text, list(self.sec_mapper.keys()), scorer=fuzz.ratio)

        # No really good sentence was found for some reason
        if score < threshold:  # arbitrary
            return None, None, score, found_sent
        else:
            sec_idx, sec_type = self.sec_mapper[found_sent]
            return sec_idx, sec_type, score, found_sent

    def get_section_type(self, extractable: Extractable) -> str:
        """
        Given a sentence, find the section type it's from.

        :param in_sentence: Sentence being queried for
        :return: string representation of the sentence type if available (e.g. "methods and materials", "results")
        """
        # Get the sentence object in which the extractable was found
        in_sentence = extractable.get_sentence()
        # Return all the section information for this sentence object
        sec_idx, section_type, score, found_sent = self.get_section_info(in_sentence)

        # Do a bit of normalization with a pre-specified list of terms mapping to canonical sections
        if section_type in results_syns:
            norm_sec_type = "results"
        elif section_type in methods_syns:
            norm_sec_type = "methods"
        elif section_type in background_syns:
            norm_sec_type = "background"
        elif section_type in discconc_syns:
            norm_sec_type = "discussion and conclusion"
        elif section_type in ["title", "subtitle"]:
            norm_sec_type = "title"
        elif section_type == "abstract":
            norm_sec_type = "abstract"
        else:
            norm_sec_type = None
        return norm_sec_type, section_type, found_sent

    def section_distance(self, context: Context, relation: Relation) -> int:
        """Calculate the distance between the sections in which the two sentences are located."""
        ctx_sentence = context.get_sentence()
        rel_sentence = relation.get_sentence()
        ctx_sec_idx, _, _, ctx_returned_sent = self.get_section_info(ctx_sentence)
        rel_sec_idx, _, _, rel_returned_sent = self.get_section_info(rel_sentence)
        if (ctx_sec_idx is None) or (rel_sec_idx is None):
            return None, None, None
        else:
            return abs(ctx_sec_idx - rel_sec_idx), ctx_returned_sent, rel_returned_sent

    def sentence_distance(self, context: Context, relation: Relation) -> int:  # maybe should genralize to extractables
        """Calculate distance between sentences that contain the Context and Relation."""
        # check if in same paper?
        # For now assuming we get sentence id when we're parsing the input annotated file.
        # True for context. Not for relations.

        # alternative version: (assumes reliable sent_idx, big assumption)
        # return abs(context.sent_idx - relation.sent_idx)

        ctx_sentence = context.get_sentence()
        rel_sentence = relation.get_sentence()
        ctx_sent_idx, _, ctx_returned_sent = self.find_sentence_idx(ctx_sentence)
        rel_sent_idx, _, rel_returned_sent = self.find_sentence_idx(rel_sentence)

        if (ctx_sent_idx is None) or (rel_sent_idx is None):
            return None, None, None
        else:
            return abs(ctx_sent_idx - rel_sent_idx), ctx_returned_sent, rel_returned_sent

    def dp_distance(self, context: Context, relation: Relation, plot=False) -> int:
        """Get distance in dependency path edges between context and relation."""
        # Get context and relation sentences
        ctx_sent = context.get_sentence()
        rel_sent = relation.get_sentence()
        edges = set()

        # Get dependency path edges for context and relation
        ctx_edges = [(u + "_ctx", v + "_ctx") for (u, v) in ctx_sent.get_edge_list()]
        rel_edges = [(u + "_rel", v + "_rel") for (u, v) in rel_sent.get_edge_list()]
        edges = edges.union(set(ctx_edges))
        edges = edges.union(set(rel_edges))

        # Finally join the two roots together as described in Noriega et al.
        ctx_rel_edge = (str(ctx_sent.get_root()).lower() + "_ctx", str(rel_sent.get_root()).lower() + "_rel")
        edges = edges.union([ctx_rel_edge])

        # Get the length and path
        graph = nx.Graph(list(edges))
        entity1 = context.get_text().lower() + "_ctx"  # MIGHT BE PROBLEM IF MANY INSTANCES OF CONTEXT WORD. THINK!
        entity2 = relation.get_main_verb().lower() + "_rel"
        dp_dist = nx.shortest_path_length(graph, source=entity1, target=entity2)

        if plot:
            plt.figure(3, figsize=(12, 12))
            pos = nx.spring_layout(graph)
            nx.draw_networkx_nodes(graph, pos, cmap=plt.get_cmap('jet'), node_size=500)
            nx.draw_networkx_labels(graph, pos)
            nx.draw_networkx_edges(graph, pos)
            plt.show()

        return dp_dist

    def num_mentions(self, in_context: Context):
        """Return number of mentions of a term in the document."""
        # Note: no ontology-based normalization or anything at this point
        mention_ct = 0
        for ctx in self.context_list:
            if ctx.get_text() == in_context.get_text():
                mention_ct += 1

        return mention_ct

    def is_closest(self, context: Context, relation: Relation):
        """
        Return whether or not input context is the closest context to the relationship.

        :param context: Context object whose text we'll check if is the closest (not sure this is best)
        :param relation: Relation object to which we're trying to find the closest context object
        :return is_closest: Boolean, is this context text the closest to the input relation?
        """
        sent_dist_list = [self.sentence_distance(cont, relation) for cont in self.context_list]
        cont_text_list = [cont.get_text() for cont in self.context_list]

        if None not in sent_dist_list:
            closest_idx = sent_dist_list.index(min(sent_dist_list))
            # Note in returning this way, it will look if ANY sentence containing the context word is the closest
            return cont_text_list[closest_idx] == context.get_text()
        else:
            return None

    def calculate_pmi(self, context: Context, relation: Relation):
        """
        Calculate section-level PMI score of a pair of concepts in one full text doc.

        :param context: Context object whose text will be used to calculate section-level PMI
        :param relation: Relation object whose two entities will be used to calculate section-level PMI
        :return ( PMI(entity1, context_word), PMI(entity2, context_word) )
        """
        pmis = []
        entity_and_conts = []
        for entity in [relation.get_entity1(), relation.get_entity2()]:
            no_entity_no_cont = 0
            entity_no_cont = 0
            cont_no_entity = 0
            entity_and_cont = 0

            for section in self.full_text_in_sections:
                section.rstrip().lower()
                found_entity = (entity in section)
                found_cont = (context.get_text() in section)
                if found_entity and found_cont:
                    entity_and_cont += 1
                elif found_entity and not found_cont:
                    entity_no_cont += 1
                elif not found_entity and found_cont:
                    cont_no_entity += 1
                elif not found_entity and not found_cont:
                    no_entity_no_cont += 1

            # TODO: Consider return values in cases of no co-occurrence
            if entity_and_cont == 0:
                pmis.append(float("-inf"))
            if (entity_and_cont + entity_no_cont) == 0:
                pmis.append(float("-inf"))
            if (entity_and_cont + cont_no_entity) == 0:
                pmis.append(float("-inf"))
            else:
                term1 = np.log2(entity_and_cont)
                term2 = np.log2((entity_and_cont + entity_no_cont) * (entity_and_cont + cont_no_entity))
                pmi_entity_cont = term1 - term2
                pmis.append(pmi_entity_cont)

            entity_and_conts.append(entity_and_cont)

        return pmis[0], entity_and_conts[0], pmis[1], entity_and_conts[1]

    def get_title(self) -> str:
        """Get title of paper."""
        return self.title

    def get_full_text(self) -> str:
        """Get full text of paper."""
        return self.full_text

    def get_full_text_sents(self):
        """Get the list of sentences in full text."""
        return self.full_text_sents

    def get_context_list(self):
        """Get list of context events associated with paper."""
        return self.context_list

    def get_relations(self):
        """Get list of relationship events associated with paper."""
        return self.relations

    def get_doi(self):
        """Get the DOI of the paper."""
        return self.doi

    def get_pmcid(self):
        """Get the PMC ID of the paper."""
        return self.pmcid

    def get_pmid(self):
        """Get the PMID of the paper."""
        return self.pmid

    def get_full_text_in_sections(self):
        """Get the list of sections of the text."""
        return self.full_text_in_sections
