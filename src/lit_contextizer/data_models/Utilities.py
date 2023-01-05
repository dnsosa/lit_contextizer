"""Various utilities functions for data processing in lit_contextizer package."""

# -*- coding: utf-8 -*-

import re
import string

import en_core_web_sm

from lit_contextizer.data_models.Extractable import Context
from lit_contextizer.data_models.Paper import Paper
from lit_contextizer.data_models.Sentence import Sentence

nlp = en_core_web_sm.load()


def drop_the_s(in_word):
    """Normalize plural with singular by looking for an s and dropping it (simple)--should catch most things."""
    return re.sub("s$", "", in_word)


def singularize_list(in_list):
    """Normalize plural list into singulars by dropping s with drop_the_s."""
    return [f"{drop_the_s(item)}" for item in in_list]


def pluralize_list(in_list):
    """Normalize mixed number items into plural forms."""
    return [f"{drop_the_s(item)}s" for item in in_list]


def singularize_and_pluralize_list(in_list):
    """Take a list of mixed number items and make sure a singular and a plural copy are present in resulting list."""
    sing_plur_list = singularize_list(in_list) + pluralize_list(in_list)
    return list(set(sing_plur_list))


def create_contexts(paper: Paper,
                    event_dict: dict,
                    max_num_sents: int = float("inf")) -> list:
    """
    Create list of context objects based on paper DOI and found events from parser.

    :param paper: paper to have contexts associated with
    :param event_dict: dictionary containing annotated context objects
    :param max_num_sents: maximum number of sentence over which to iterate to look for contexts -- just for debugging
    :return: list of context objects
    """
    # This context identifier will help index context objects at paper level.
    uid_counter = 0
    context_list = []

    # Iterate through sentences, find instances where 1-, 2-, or 3-grams match an event in the event dict
    for sent_idx, sent in enumerate(paper.get_full_text_sents()):
        word_list = sent.split(" ")
        for word_idx, word in enumerate(word_list):
            # strip trailing punctuation like ',' '.'
            word = word.translate(str.maketrans('', '', string.punctuation))
            match = None
            two_mer = None
            three_mer = None

            curr_tok = word.lower()
            one_mer = curr_tok

            # Look ahead to construct bi- and trigrams
            if word_idx < len(word_list) - 1:
                next_tok = word_list[word_idx + 1].lower()
                two_mer = f"{curr_tok} {next_tok}"

                if word_idx < len(word_list) - 2:
                    nextnext_tok = word_list[word_idx + 2].lower()
                    three_mer = f"{curr_tok} {next_tok} {nextnext_tok}"

            # Check if this ngram is in the event dictionary derived from XML tags
            if one_mer in event_dict:
                match = one_mer
            if two_mer in event_dict:
                match = two_mer
            if three_mer in event_dict:
                match = three_mer

            # Create a new context object
            if match is not None:
                attributes = event_dict[match]
                uuid = None
                if "uuid" in attributes.keys():
                    uuid = attributes["uuid"]

                new_cont = Context(uuid=uuid,
                                   ctx_type=attributes["type"],
                                   ctx_uid=uid_counter,
                                   text=drop_the_s(match),
                                   paper_doi=paper.get_doi(),
                                   paper_pmcid=paper.get_pmcid(),
                                   paper_pmid=paper.get_pmid(),
                                   start_idx=attributes["pos"] - len(match),
                                   end_idx=attributes["pos"],
                                   sent_idx=sent_idx,
                                   sentence=Sentence(sent))

                context_list.append(new_cont)
                uid_counter += 1

        # Just for debugging!
        if sent_idx >= max_num_sents:
            break

    # Now associate this collection of context objects with the paper object
    paper.set_context_list(context_list)
    return None
