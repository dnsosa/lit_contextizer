"""Utilities for extracting features from papers in a more space efficient way."""

# -*- coding: utf-8 -*-

import errno
import os
from os import path

import bioc

from lit_contextizer.data_models.Extractable import Context, Extractable, Relation
from lit_contextizer.data_models.Paper import Paper
from lit_contextizer.data_models.Utilities import drop_the_s
from lit_contextizer.data_models.constants import BACKGROUND_SYNS_PATH, DISCCONC_SYNS_PATH, METHODS_SYNS_PATH, \
    RESULTS_SYNS_PATH

import numpy as np

import pandas as pd

from rapidfuzz import fuzz, process

local_biocxml_dir = "/Users/dnsosa/Desktop/AltmanLab/bai/biotext/full_texts/PM_files"

background_syns = set(pd.read_csv(BACKGROUND_SYNS_PATH).synonyms)
methods_syns = set(pd.read_csv(METHODS_SYNS_PATH).synonyms)
results_syns = set(pd.read_csv(RESULTS_SYNS_PATH).synonyms)
discconc_syns = set(pd.read_csv(DISCCONC_SYNS_PATH).synonyms)


def extract_features(paper_pile,
                     do_calculate_pmi=False,
                     do_calculate_in_mesh=False,
                     mesh_headings_in_pmc=False,
                     biocxmls_pmc_dir=local_biocxml_dir,
                     biocxmls_pubmed_dir=local_biocxml_dir,
                     annotated_connects=None,
                     no_cell_line=True,
                     stop_count=float("inf"),
                     is_enrique=False):
    """
    Extract features by loading full texts, etc.

    :param paper_pile: paper pile (dictionary) from which to extract features
    :param do_calculate_pmi: if True, extract/calculate PMI information. Useful for not PMI'ing val
    :param do_calculate_in_mesh: if True, extract info about MeSH headings
    :param mesh_headings_in_pmc: if True, MeSH headings are expected to be found in the PMC file (not just PMID)
    :param biocxmls_pmc_dir: directory where the relevant full text files are located
    :param biocxmls_pubmed_dir: directory where the relevant pubmed (abstract only) files are located
    :param annotated_connects: if not None, expect list of tuples (paper_id, rel_sent, con_word) that were in annots
    :param no_cell_line: if True, do not include contexts labeled as CellLine (noisy) for extracting features
    :param stop_count: max number of features to extract
    :param is_enrique: flag if we're working with the ENA corpus
    :return: DataFrame of all extracted features
    """
    ct = 0

    # Build pandas DF here
    rows = []

    for paper_id in paper_pile:
        ct += 1
        paper = paper_pile[paper_id]
        if (len(paper.get_context_list()) > 0) and (len(paper.get_relations()) > 0):

            # Extract section info from full text .biocxml files
            # Note this conditional gets the right files in the cases of val extractions I believe

            if is_enrique:
                home_dir = "/Users/dnsosa/Desktop/AltmanLab/bai/biotext/full_texts"
                biocxml_file = os.path.join(home_dir, f"{paper.get_pmcid()}.biocxml")
            else:
                biocxml_file = os.path.join(biocxmls_pmc_dir, f"{paper.get_pmcid()}.biocxml")

            if path.exists(biocxml_file):
                full_text_in_sections = []
                sec_mapper = {}
                with open(biocxml_file, 'rb') as f:

                    collection = bioc.biocxml.load(f)
                    document = collection.documents[0]

                    for sec_idx, sec in enumerate(document.passages):
                        full_text_in_sections.append(sec.text)
                        # Get the name of the section
                        if sec.infons['section'] != 'article':  # 'article' = body of text, we want title/abstract
                            sec_type = sec.infons['section']
                        else:  # get the name of the section e.g. methods, results
                            sec_type = sec.infons['subsection']
                        sec_text_remove_extra_periods = sec.text.replace("et al.", "et al").replace("ig. ", "ig ")
                        sent_list = [s + "." for s in sec_text_remove_extra_periods.split(". ") if s]
                        for sent in sent_list:
                            sec_mapper[sent] = (sec_idx, sec_type)

            else:
                # Don't even consider papers without that biocxml file
                continue

            # Now extract features. Maintain dictionaries of dictionaries for quick indexing if needed
            rc_sent_dist = {}  # Dictionary of dictionaries
            rc_sec_dist = {}  # Dictionary of dictionaries -- this model may change
            if do_calculate_pmi:
                rc_pmi_infos = {}  # same
            r_closest_c = {}  # same

            # These are used for memoization
            sent_sent_idxs = {}
            sent_sec_info = {}
            con_fps = {}
            con_in_mesh_headings = {}

            # Build the Pandas DF here

            for rel in paper.get_relations():
                sent_dists = {}
                sec_dists = {}
                pmi_infos = {}
                is_closests = {}
                for con in paper.get_context_list():
                    if no_cell_line and (con.get_ctx_type() == "CellLine"):
                        continue

                    # Build single rel-con pair of DF
                    row = {'paper_id': paper_id,
                           'rel': rel.get_text(),
                           'con': con.get_text(),
                           'con_sent': con.get_sentence().get_text(),
                           'con_type': con.get_ctx_type(),
                           'ent_1': rel.get_entity1(),
                           'ent_2': rel.get_entity2()}

                    # Sentence distances
                    con_sent_idx, sent_sent_idxs = get_sent_idx(con, paper.get_full_text_sents(), sent_sent_idxs)
                    rel_sent_idx, sent_sent_idxs = get_sent_idx(rel, paper.get_full_text_sents(), sent_sent_idxs)
                    if (rel_sent_idx is not None) and (con_sent_idx is not None):
                        sent_distance = abs(rel_sent_idx - con_sent_idx)
                        sent_dists[con] = sent_distance
                    else:
                        sent_distance = None
                        sent_dists[con] = None
                    row['sent_dist'] = sent_distance

                    # Section distances and section info
                    con_sec_idx, con_sec, norm_con_sec, sent_sec_info = get_sec_info(con, sec_mapper, sent_sec_info)
                    rel_sec_idx, rel_sec, norm_rel_sec, sent_sec_info = get_sec_info(rel, sec_mapper, sent_sec_info)
                    if (rel_sec_idx is not None) and (con_sec_idx is not None):
                        sec_distance = abs(rel_sec_idx - con_sec_idx)
                        sec_dists[con] = sec_distance
                    else:
                        sec_distance = None
                        sec_dists[con] = None
                    row['sec_dist'] = sec_distance
                    row['rel_sec'] = rel_sec
                    row['norm_rel_sec'] = norm_rel_sec
                    row['con_sec'] = con_sec
                    row['norm_con_sec'] = norm_con_sec

                    # Number of mentions of cont word
                    row['num_con_mentions'] = num_mentions(paper, con)

                    # Is con sentence in FP
                    fp, con_fps = is_fp(con, con_fps)
                    row['is_con_fp'] = fp

                    # PMI info
                    if do_calculate_pmi:
                        pmi_info = calculate_pmi(con, rel, full_text_in_sections)  # note the format of the output
                        pmi_infos[con] = pmi_info
                        row['pmi_1'] = pmi_info[0]
                        row['num_sec_cooccur_ent_1'] = pmi_info[1]
                        row['pmi_2'] = pmi_info[2]
                        row['num_sec_cooccur_ent_2'] = pmi_info[3]

                    if do_calculate_in_mesh:
                        if con.get_text() in con_in_mesh_headings:
                            row['con_in_mesh_headings'] = con_in_mesh_headings[con.get_text()]

                        else:
                            # TODO: Consider when to retrieve PMID
                            if mesh_headings_in_pmc:
                                print(biocxml_file)
                                mesh_containing_file = biocxml_file
                            else:
                                my_pmid = document.infons['pmid']

                                if os.path.isfile(os.path.join(biocxmls_pubmed_dir, f"{my_pmid}.biocxml")):
                                    pmid_file = os.path.join(biocxmls_pubmed_dir, f"{my_pmid}.biocxml")
                                elif os.path.isfile(os.path.join(biocxmls_pubmed_dir, f"PM{my_pmid}.biocxml")):
                                    pmid_file = os.path.join(biocxmls_pubmed_dir, f"PM{my_pmid}.biocxml")
                                else:
                                    print(f"File not for pmid {my_pmid}.biocxml or PM{my_pmid}.biocxml in directory "
                                          f"{biocxmls_pubmed_dir}.")
                                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                                            os.path.join(biocxmls_pubmed_dir, f"{my_pmid}.biocxml"))

                                mesh_containing_file = pmid_file

                            with open(mesh_containing_file, 'rb') as f:
                                collection = bioc.biocxml.load(f)
                                doc = collection.documents[0]

                                if 'meshHeadings' in doc.infons:
                                    mesh_headings = doc.infons['meshHeadings']
                                    if mesh_headings is None:
                                        row["con_in_mesh_headings"] = None
                                        con_in_mesh_headings[con.get_text()] = None
                                    else:
                                        row['con_in_mesh_headings'] = (con.get_text() in mesh_headings.lower())
                                        con_in_mesh_headings[con.get_text()] = (con.get_text() in mesh_headings.lower())
                                else:
                                    row['con_in_mesh_headings'] = None
                                    con_in_mesh_headings[con.get_text()] = None

                    # Annotated info in case of val
                    if annotated_connects is not None:
                        rel_text = rel.get_text().rstrip()
                        con_text = con.get_text().lower()
                        row['annotation'] = ((paper_id, rel_text, con_text) in annotated_connects)

                    rows.append(row)

                # Update the dictionaries
                rc_sent_dist[rel] = sent_dists
                rc_sec_dist[rel] = sec_dists
                if do_calculate_pmi:
                    rc_pmi_infos[rel] = pmi_infos

                # After one iteration, calculate the closest cont to rel thing
                sent_dists_no_nones = [val for val in sent_dists.values() if val]
                if not sent_dists_no_nones:  # empty list
                    min_sent_dist = float("inf")
                else:
                    min_sent_dist = min(sent_dists.values())
                closest_terms = [k.get_text() for k, v in sent_dists.items() if v == min_sent_dist]  # allows for ties

                for con in paper.get_context_list():
                    is_closests[con] = (con.get_text() in closest_terms)

                r_closest_c[rel] = is_closests
                # Note, presently not doing anything with those dictionaries of dictionaries.

            if ct == stop_count:
                break

        if ct % 100 == 0:
            print(f"Extracted features from {ct} papers (of 21243)")

    df = pd.DataFrame(rows)

    # Calculate features based on number of mentions
    # Calculate proportion of mentions
    df_mentions_grp = df.groupby(['rel'])['num_con_mentions']
    df['con_mention_frac'] = df['num_con_mentions'] / df_mentions_grp.transform('sum')
    df['con_mention_50'] = df['con_mention_frac'] >= 0.5

    # Calculate is maximum context by count
    df = df.assign(num_con_mentions_max=df_mentions_grp.transform(max))
    df["is_con_mention_max"] = (df['num_con_mentions'] == df['num_con_mentions_max'])

    # Finally, calculate "is_closest", which is facilitated by Pandas DF groupby
    print(df.columns)
    df_grp = df.groupby(['rel'])['sent_dist']
    df = df.assign(min_sent_dist=df_grp.transform(min))
    df["is_closest_cont_by_sent"] = (df['sent_dist'] == df['min_sent_dist'])

    return df


def num_mentions(paper: Paper, in_context: Context):
    """
    Return number of mentions of a term in the document.

    :param paper: input paper to search for number of mentions
    :param in_context: context to find occurrences of
    :return: number of mentions
    """
    # Note: no ontology-based normalization or anything at this point
    mention_ct = 0
    for ctx in paper.get_context_list():
        if drop_the_s(ctx.get_text()) == drop_the_s(in_context.get_text()):  # Drop the s, double checking
            mention_ct += 1

    return mention_ct


def get_sent_idx(extractable: Extractable, text_in_sents, sent_sent_idxs, threshold=80):
    """
    Get sentence index of an extractable in the full text in sentences using memoization for efficiency.

    :param extractable: extractable to find the sentence index of
    :param text_in_sents: full text in a list of sentences
    :param sent_sent_idxs: current state of sentence indexes dictionary
    :param threshold: QC threshold below which no high quality match was found (0-100)
    :return: (sentence index if in the full text, sent_sent_idxs dictionary updated)
    """
    in_sent = extractable.get_sentence().get_text()
    if in_sent not in sent_sent_idxs.keys():
        found_sent, score, idx = process.extractOne(in_sent, text_in_sents, scorer=fuzz.ratio)
        if score < threshold:
            idx = None
        sent_sent_idxs[in_sent] = idx

    return sent_sent_idxs[in_sent], sent_sent_idxs


def get_sec_info(extractable: Extractable, sec_mapper, sent_sec_info, threshold=80):
    """
    Get sentence info of an extractable in the full text in sentences, memoize the results.

    :param extractable: extractable to find the sentence index of
    :param sec_mapper: maps the sentence to the section info based on the full text document
    :param sent_sec_info: current state of sentence to section info (name of section and index)
    :param threshold: QC threshold below which no high quality match was found (0-100)
    :return: (section index if in the full text, section name, normalized section name, memoized information updated)
    """
    in_sent = extractable.get_sentence().get_text()
    if in_sent not in sent_sec_info.keys():
        found_sent, score, _ = process.extractOne(in_sent, sec_mapper.keys(), scorer=fuzz.ratio)
        if score < threshold:
            idx = None
            sec_name = None
        else:
            idx, sec_name = sec_mapper[found_sent]

        sent_sec_info[in_sent] = (idx, sec_name)

    else:
        idx, sec_name = sent_sec_info[in_sent]

    # Do a bit of normalization with a pre-specified list of terms mapping to canonical sections
    if sec_name in results_syns:
        norm_sec_name = "results"
    elif sec_name in methods_syns:
        norm_sec_name = "methods"
    elif sec_name in background_syns:
        norm_sec_name = "background"
    elif sec_name in discconc_syns:
        norm_sec_name = "discussion and conclusion"
    elif sec_name in ["title", "subtitle"]:
        norm_sec_name = "title"
    elif sec_name == "abstract":
        norm_sec_name = "abstract"
    else:
        norm_sec_name = None

    return idx, sec_name, norm_sec_name, sent_sec_info


def are_both_in_results(con_sec, rel_sec):
    """Return if both things are in results."""
    return (con_sec == "results") and (rel_sec == "results")


def is_fp(con, con_fps):
    """Return if the context object is in first person, memoize the results."""
    if con not in con_fps.keys():
        sent = con.get_sentence()
        pov = sent.get_pov()
        con_fps[con] = ("1" in pov)

    return con_fps[con], con_fps


def calculate_pmi(context: Context, relation: Relation, full_text_in_secs):
    """
    Calculate section-level PMI score of a pair of concepts in one full text doc.

    :param context: Context object whose text will be used to calculate section-level PMI
    :param relation: Relation object whose two entities will be used to calculate section-level PMI
    :param full_text_in_secs: representation of the full text in sections, a comma separated list of section text
    :return ( PMI(entity1, context_word), PMI(entity2, context_word) )
    """
    pmis = []
    entity_and_conts = []
    for entity in [relation.get_entity1(), relation.get_entity2()]:
        no_entity_no_cont = 0
        entity_no_cont = 0
        cont_no_entity = 0
        entity_and_cont = 0

        for section in full_text_in_secs:
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

        # Try to catch cases when PMI is undefined
        if entity_and_cont == 0:
            pmis.append(float("-inf"))
        if (entity_and_cont + entity_no_cont) == 0:
            pmis.append(float("-inf"))
        if (entity_and_cont + cont_no_entity) == 0:
            pmis.append(float("-inf"))
        else:
            pmi_entity_cont = np.log2(entity_and_cont) - np.log2(
                (entity_and_cont + entity_no_cont) * (entity_and_cont + cont_no_entity))
            pmis.append(pmi_entity_cont)

        entity_and_conts.append(entity_and_cont)

    return pmis[0], entity_and_conts[0], pmis[1], entity_and_conts[1]


def in2str(in_data: str):
    """Convert input malformatted PMIDs (sometimes tuple, sometimes string)."""
    if type(in_data) == "tuple":
        return str(in_data[0])
    elif type(in_data) == "int":
        return str(in_data)
    else:
        return in_data
