"""Various utilities functions for data processing in lit_contextizer package."""

# -*- coding: utf-8 -*-

import re
import string

import en_core_web_sm

from lit_contextizer.data_models.Extractable import Context
from lit_contextizer.data_models.MLStripper import MLStripper
from lit_contextizer.data_models.Paper import Paper
from lit_contextizer.data_models.Sentence import Sentence
from lit_contextizer.data_models.constants import ALL_GROUNDINGS_PATH

import numpy as np

import pandas as pd

nlp = en_core_web_sm.load()


def two_common(a, b, c):
    """
    At least two annotators agreed on the annotation.

    :param a: annotator A's annotations
    :param b: annotator B's annotations
    :param c: annotator C's annotations
    :return: the majority consensus if any
    """
    if (a == b) and (a != "nan"):
        return a
    elif (b == c) and (b != "nan"):
        return b
    elif (c == a) and (c != "nan"):
        return c
    else:
        return None


def load_all_groundings(all_groundings_file: str = ALL_GROUNDINGS_PATH):
    """
    Load groundings from the ENA dataset.

    :param all_groundings_file: filename of groundings document
    :return: grounding mappers in both directions
    """
    # Creation dictionaries of maps in both directions
    onto2texts = {}
    text2ontos = {}

    with open(all_groundings_file, 'r') as f:
        for line in f:
            line_list = line.split("\t")
            onto_term = line_list[0]
            text_terms = line_list[1].rstrip().split(', ')
            for term in text_terms:
                if onto_term not in onto2texts:
                    onto2texts[onto_term] = [term.lower()]
                else:
                    if term.lower() not in onto2texts[onto_term]:
                        onto2texts[onto_term].append(term.lower())

                if term not in text2ontos:
                    text2ontos[term.lower()] = [onto_term]
                else:
                    if onto_term not in text2ontos[term.lower()]:
                        text2ontos[term.lower()].append(onto_term)

    return onto2texts, text2ontos


def generate_pairs_features_df(paper_dict: dict):
    """
    Generate DF of pairs of (con, rel) from a paper pile with corresponding features to be used for classification.

    :param paper_dict: A paper pile from which to generate all pairs of (cont, rel) and extract some features
    :return: data frame of relation by contexts with distances
    """
    dois = []
    rel_sens = []
    con_sens = []
    con_word = []
    sens_dists = []
    num_mentions = []

    for paper_doi in paper_dict.keys():
        paper = paper_dict[paper_doi]

        paper_context_list = paper.get_context_list()
        paper_relation_list = paper.get_relations()
        if len(paper_context_list) > 0:
            for rel in paper_relation_list:
                for con in paper_context_list:
                    dist = paper.sentence_distance(con, rel)
                    dois.append(paper.get_doi())
                    rel_sens.append(rel.get_sentence().get_text())
                    con_sens.append(con.get_sentence().get_text())
                    con_word.append(con.get_text())
                    sens_dists.append(dist)
                    num_mentions.append(paper.num_mentions(con))

    df = pd.DataFrame(list(zip(dois, rel_sens, con_sens, con_word, sens_dists, num_mentions)),
                      columns=['DOI', 'RelationSent', 'ContextSent', 'ContextWord', 'SentDist',
                               "NumMentions"]).drop_duplicates()

    return df


def calculate_pmi_jake_txt(x: str, y: str, in_file: str):
    """
    Calculate section-level PMI score of a pair of concepts in one full text doc.

    :param x: First concept, a strip to be matched
    :param y: Second concept, also a string to be matched
    :param in_file: File path containing full text of document. NOTE expecting that newlines demarcate sections!
    :return: PMI of the two concepts in a single document
    """
    no_x_no_y = 0
    x_no_y = 0
    y_no_x = 0
    x_and_y = 0

    with open(in_file, 'r') as f:
        for line in f:
            if line == "\n":
                continue
            line = line.rstrip().lower()
            found_x = (x in line)
            found_y = (y in line)
            if found_x and found_y:
                x_and_y += 1
            elif found_x and not found_y:
                x_no_y += 1
            elif not found_x and found_y:
                y_no_x += 1
            elif not found_x and not found_y:
                no_x_no_y += 1

    pmi_11 = np.log2(x_and_y) - np.log2((x_and_y + x_no_y) * (x_and_y + y_no_x))

    return pmi_11


def remove_non_ascii(s: str):
    """
    Remove non-ascii characters from a string.

    :param s: input string
    :return: cleaned string
    """
    return "".join(i for i in s if (ord(i) < 128 or ord(i) == 36))


def strip_tags(html: str):
    """
    Strip tags from an XML-like marked up text.

    :param html: input html string
    :return: cleaned string
    """
    s = MLStripper()
    s.feed(html)
    stripped = s.get_data()
    return stripped


def fix_xml(in_str: str):
    """
    Escape annoying characters in XML using manual regex rules. TODO: Get pre-escaped input data.

    :param in_str: input string
    :return: parsed string with annoying characters escaped
    """
    # parsed_in_str = re.sub(r'\&', r'\&amp;', in_str)
    # parsed_in_str = re.sub(r'\<\s', r'\&lt; ', parsed_in_str)
    # parsed_in_str = re.sub(r'\<(?=[0-9])', r'\&lt; ', parsed_in_str)  # P < .05 business
    # parsed_in_str = re.sub(r'\<(?=.[0-9])', r'\&lt; ', parsed_in_str)  # negative numbers
    # parsed_in_str = re.sub(r'\<(?=.[0-9])', r'\&lt; ', parsed_in_str)  # negative numbers
    # # parsed_in_str = re.sub('(?<=[A-Z])\<(?!\\\\)','\&lt; ', parsed_in_str) #wrong, less specific same case of K<K
    # parsed_in_str = re.sub(r'(?<=K)\<K', r'K\&lt;K ', parsed_in_str)  # very specific
    # parsed_in_str = re.sub(r'\<namespace reference\>:\<entity reference\>', '', parsed_in_str)  # very specific
    # parsed_in_str = re.sub(r'\<\<', r'\<', parsed_in_str)  # malformatted <<entity tag
    # parsed_in_str = re.sub(r'(?<=\s0)\<', r'\&lt;', parsed_in_str)  # number before (like latex)

    # Ampersand
    parsed_in_str = re.sub(r'&', r'\&amp;', in_str)
    # Less than tag
    parsed_in_str = re.sub(r'<(?!entity)(?!\/entity)(?!root)(?!\/root)', r'\&lt;', parsed_in_str)
    # Greater than tag
    gt_regex = r'>(?<!<\/entity>)(?<!root>)(?<!entity reference>)(?=[^<]*<entity[^<>]*>)'
    parsed_in_str = re.sub(gt_regex, r'\&gt;', parsed_in_str)
    # Remove entity reference tags
    parsed_in_str = re.sub(r'<entity reference>', '', parsed_in_str)
    return parsed_in_str


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
