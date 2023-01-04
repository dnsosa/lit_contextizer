"""Object for loading input data and creating relevant paper and extractables objects."""

# -*- coding: utf-8 -*-

import io
import json
import os
import re
from functools import reduce
from xml.sax import SAXParseException, make_parser  # noqa: S406

import bioc

from lit_contextizer.data_models.Extractable import Relation
from lit_contextizer.data_models.Paper import Paper
from lit_contextizer.data_models.PaperUtilities import extract_features, in2str
from lit_contextizer.data_models.Sentence import Sentence
from lit_contextizer.data_models.Utilities import create_contexts, drop_the_s, fix_xml, load_all_groundings, \
    singularize_and_pluralize_list, two_common
from lit_contextizer.data_models.Xml2PseudoJson import Xml2PseudoJson
from lit_contextizer.data_models.constants import BIOCXML_DIR_DEFAULT, BIOCXML_OUT_DIR_DEFAULT, \
    CON_TERMS_LEXICON_PATH, CT_TERMS_LEXICON_PATH, FULL_TEXTS_FILE, RELATIONSHIPS_FILE, TISSUE_TERMS_LEXICON_PATH


import pandas as pd

# Suppress warnings
pd.options.mode.chained_assignment = None


class DataLoader:
    """This object stores representations of papers, extracted events, and extracted relationships."""

    def __init__(self):
        """
        Create a DataLoader object. Note currently the relationships and paper_pile attributes are redundant.

        TODO: Revisit redundancy?
        """
        self.events = {}
        self.relationships = {}
        self.paper_pile = {}
        self.ena_paper_pile = {}
        self.global_section_mapper = {}
        self.all_pubmed_relations_df = None
        self.annot_id2entrez = {}
        self.annot_id2text = {}
        self.pmid2contexts = {}
        self.contexts2pmid = {}
        self.pmid2species = {}
        self.pmid2speciesText = {}
        self.pubmed_ppi_paper_pile = {}
        self.pubmed_ppi_features_df = None

    def parse_pubmed_full_texts(self,
                                in_dir: str = BIOCXML_DIR_DEFAULT,
                                load_max=float("inf")):
        """
        Load the BioCXML file containing annotated PubMed files provided by Jake.

        :param in_dir: input directory containing BioCXML files with groups of annotated pubmed papers
        :param load_max: max number of papers to load
        """
        df_list = []  # list to become DF
        total_paper_count = 0

        # Try the default dir in Sherlock if none provided
        if in_dir is None:
            in_dir = BIOCXML_DIR_DEFAULT

        for in_file in sorted(os.listdir(in_dir)):
            if not in_file.startswith("pmc"):
                continue

            print(os.path.join(in_dir, in_file))

            with open(os.path.join(in_dir, in_file), 'rb') as f:

                parser = bioc.biocxml.load(f)

                for doc in parser.documents:

                    pmid = in2str(doc.infons["pmid"])
                    pmcid = in2str(doc.infons["pmcid"])

                    for sec in doc.passages:
                        for annot in sec.annotations:
                            annot_type = annot.infons["type"]  # Chemical, Gene, xref, Species, Cell Context
                            if annot_type == "Gene":
                                self.annot_id2entrez[annot.id] = annot.infons["conceptid"]
                                self.annot_id2text[annot.id] = annot.text
                            elif annot_type == "CellContext":
                                context = in2str(annot.infons["normalized"]).lower()
                                if pmid not in self.pmid2contexts:
                                    self.pmid2contexts[pmid] = {context}
                                else:
                                    self.pmid2contexts[pmid].add(context)

                                if context not in self.contexts2pmid:
                                    self.contexts2pmid[context] = {pmid}
                                else:
                                    self.contexts2pmid[context].add(pmid)
                            elif annot_type == "Species":
                                species = in2str(annot.infons["conceptid"])
                                if pmid not in self.pmid2species:
                                    self.pmid2species[pmid] = {species}
                                    self.pmid2speciesText[pmid] = {annot.text}
                                else:
                                    self.pmid2species[pmid].add(species)
                                    self.pmid2speciesText[pmid].add(annot.text)

                        if len(sec.relations) > 0:
                            for rel in sec.relations:
                                sen = rel.infons["formatted_sentence"]
                                e1_refid = rel.nodes[0].refid
                                e2_refid = rel.nodes[1].refid

                                parsed_entities = re.findall(r'<entity[0-9]>(.+?)</entity[0-9]>', sen)
                                if len(parsed_entities) == 2:
                                    row = {"rel": sen,
                                           "pmid": pmid,
                                           "pmcid": pmcid,
                                           "entity1": e1_refid,
                                           "entity2": e2_refid,
                                           "entity1_text": parsed_entities[0],
                                           "entity2_text": parsed_entities[1],
                                           "filename": in_file}
                                else:
                                    print(f"TOO MANY ENTITIES PARSED IN SENTENCE: {sen}")

                                if 'distance_to_nearest_sentence_with_citation' in rel.infons:
                                    row["distance_to_nearest_sentence_with_citation"] =\
                                        rel.infons['distance_to_nearest_sentence_with_citation']

                                df_list.append(row)

                total_paper_count += 1
                if total_paper_count == load_max:
                    break

            if total_paper_count == load_max:
                break

        self.all_pubmed_relations_df = pd.DataFrame(df_list)
        self.all_pubmed_relations_df["entity1_entrez"] = self.all_pubmed_relations_df["entity1"].\
            map(self.annot_id2entrez)
        self.all_pubmed_relations_df["entity1_entrez"] = self.all_pubmed_relations_df["entity1_entrez"].\
            apply(lambda x: x.split(';'))
        self.all_pubmed_relations_df = self.all_pubmed_relations_df.explode("entity1_entrez").reset_index(drop=True)

        self.all_pubmed_relations_df["entity2_entrez"] = self.all_pubmed_relations_df["entity2"].\
            map(self.annot_id2entrez)
        self.all_pubmed_relations_df["entity2_entrez"] = self.all_pubmed_relations_df["entity2_entrez"].\
            apply(lambda x: x.split(';'))
        self.all_pubmed_relations_df = self.all_pubmed_relations_df.explode("entity2_entrez").reset_index(drop=True)

        def make_keys(x):
            sorted_entrez = sorted([x['entity1_entrez'], x['entity2_entrez']])  # Alphabetically NOT numerically sorted
            return '_'.join(sorted_entrez)

        self.all_pubmed_relations_df['combined_entities_key'] = self.all_pubmed_relations_df.apply(make_keys, axis=1)

        # Remove XML tags on protein/gene entities
        self.all_pubmed_relations_df['rel'] = self.all_pubmed_relations_df['rel'].str.replace('<[^<]+>', "",
                                                                                              regex=True)
        return self.all_pubmed_relations_df

    def query_context_specific_ppis(self,
                                    context_list: list,
                                    ppi_network_path: str,
                                    min_conf: float = 0):
        """
        Query papers containing specific PPIs and context mentions.

        :param context_list: list of contexts to query
        :param ppi_network_path: file path of ppi network
        :param min_conf: minimum confidence score for PPIs
        """
        def make_keys2(x):
            sorted_entrez = sorted([str(int(x['Gene1'])), str(int(x['Gene2']))])  # alphabet sorted, NOT like an int
            return '_'.join(sorted_entrez)

        ppinet_df = pd.read_csv(ppi_network_path, sep='\t')
        ppinet_df.columns = ["Gene1", "Gene2", "Conf"]
        print(f"Input PPI network has {len(ppinet_df)} edges")
        ppinet_df = ppinet_df[ppinet_df.Conf >= min_conf]
        print(f"After filtering to min. confidence level {min_conf}, PPI network has {len(ppinet_df)} edges")

        ppinet_df["query_genes"] = ppinet_df.apply(make_keys2, axis=1)

        retrieved_papers = set()
        for context in self.contexts2pmid.keys():
            for query_context in context_list:
                if query_context in context:  # Like a substring
                    pmids = self.contexts2pmid[context]
                    for pmid in pmids:
                        retrieved_papers.add(str(pmid))

        pmid_results_df = pd.DataFrame(list(retrieved_papers), columns=['pmid'])

        # Now for the filtration
        relations_df_pmid_filtered = self.all_pubmed_relations_df.merge(pmid_results_df, on="pmid", how="inner")
        relations_df_pmid_genes_filtered = relations_df_pmid_filtered.merge(ppinet_df,
                                                                            left_on="combined_entities_key",
                                                                            right_on="query_genes",
                                                                            how="inner")
        return relations_df_pmid_genes_filtered
        # Still need to now be able to process and extract features from those papers in the paper pile.......

    def parse_annotated_full_texts(self,
                                   in_file: str = FULL_TEXTS_FILE,
                                   load_max=float("inf"),
                                   cell_type_only: bool = False):
        """
        Load the JSON file containing the full annotated texts and create corresponding objects.

        :param in_file: input JSON file containing annotations
        :param load_max: input for determining number of papers to load--strictly for debugging purposes
        :param in_keys: set of keys for which to parse annotated full text and extract context mentions
        :param cell_type_only: if True, only extract contexts that are of type cell type.
        """
        # Read in the raw records
        with open(in_file) as doc_json_file:
            count = 0

            # For each file, create a new paper object
            for line in doc_json_file:
                record = json.loads(line)

                if count % 1000 == 0:
                    print(f"{count} full texts read (of 21243)")

                # idk why sometimes there's no PMC ID
                if 'pmc' not in record.keys():
                    print(f"No PMC for record {count}")
                    continue

                if 'title' not in record['plain_text'].keys():
                    print(f"No title for record {record['pmc']}")
                    continue

                if 'abstract' not in record['plain_text'].keys():
                    print(f"No abstract for record {record['pmc']}")
                    continue

                if 'text' not in record['plain_text'].keys():
                    print(f"No text for record {record['pmc']}")
                    continue

                # TODO: Need to remove the PM at the beginning
                pmid = record['doc_id'].split("|")[0][2:]

                paper = Paper(title=record['plain_text']['title'],
                              abstract=record['plain_text']['abstract'],
                              full_text=record['plain_text']['text'],
                              pmcid=record['pmc'],
                              pmid=pmid,
                              doi=record['doi'],
                              journal=record['journal'])

                count += 1

                # Parse annotated full text or abstract, whatever's available. Make it a well-formatted XML with tags.
                if 'text' in record['annotated_text'].keys():
                    in_str = f"<root>{record['annotated_text']['text']}</root>"
                elif 'abstract' in record['annotated_text'].keys():
                    in_str = f"<root>{record['annotated_text']['abstract']}</root>"

                # Set up parser for handling XML
                parser = make_parser()  # noqa: S317
                handler = Xml2PseudoJson()
                parser.setContentHandler(handler)
                parsed_in_str = fix_xml(in_str)
                in_xml = io.StringIO(parsed_in_str)

                try:
                    # Parse the XML -- resulting dictionary of events is now in handler.event_dict
                    parser.parse(in_xml)

                    if cell_type_only:
                        just_ct_events = {}
                        for event in handler.event_dict:
                            if handler.event_dict[event]["type"] in ["CellType", "CellLine", "Tissue"]:
                                just_ct_events[event] = handler.event_dict[event]
                        create_contexts(paper, just_ct_events)
                    else:
                        # Create context objects by cross-referencing document with extracted events dictionary
                        create_contexts(paper, handler.event_dict)

                    # Store this event dictionary and paper
                    self.events[record['pmc']] = handler.event_dict
                    self.paper_pile[record['pmc']] = paper

                except SAXParseException as exc:
                    # Likely from superfluous unescaped character. This is for debugging and creating regexes.
                    print("Malformatted XML!")  # noqa: T001
                    # width = 50
                    # loc = 31683
                    # print(in_str[loc - width:loc + width])
                    # print(parsed_in_str[loc - width:loc + width])
                    print(in_str)
                    raise exc
                    break

                if count > load_max:
                    # Limit number of files loaded for debugging.
                    break

        return None

    def parse_relationships_file(self, in_file: str = RELATIONSHIPS_FILE, load_max=float("inf")):
        """
        Load the relationships file. This file contains protein-protein interactions and indications of polarity.

        :param in_file: input .csv file
        :param load_max: input for determining number of papers to load--strictly for debugging purposes
        """
        # Load file of relations
        svo_contra_df = pd.read_csv(in_file, index_col=[0])

        # Iterate over relations and associate them with the proper paper object
        count = 0
        for _index, row in svo_contra_df.iterrows():
            pmcid = f"PMC{row['pmcid']}"
            pmid = f"PM{row['pmid']}"
            relation = Relation(main_verb=row['predicate'],
                                entity1=row["entity1_standardised"],
                                entity2=row["entity2_standardised"],
                                text=row['text'],
                                paper_doi=row['doi'],
                                paper_pmcid=pmcid,
                                paper_pmid=pmid,
                                start_idx=row['entity1_begin_index'],
                                end_idx=row['entity2_end_index'],
                                sent_idx=row['sentence_index'],
                                sentence=Sentence(row['text']))  # NOTE This index might not agree with our indices

            # If we've found a new paper, create a new object. Presumably this doesn't happen since the set of
            # papers in the annotated set is larger? Might not be the case because of PMC vs Elsevier etc.
            if relation.get_paper_pmcid() not in self.paper_pile:
                paper = Paper(title=row['doc_title_string'],
                              abstract=None,
                              full_text=None,
                              doi=row['doi'],
                              pmcid=pmcid,
                              pmid=pmid,
                              journal=row['journal_string'])
                self.paper_pile[pmcid] = paper

            self.paper_pile[pmcid].add_relation(relation)

            # Add the relation to the relation list. Again this feature might be redundant.
            if pmcid in self.relationships:
                self.relationships[pmcid].append(relation)
            else:
                self.relationships[pmcid] = [relation]

            if count > load_max:
                break

            count += 1

        return None

    def parse_annotation_files_for_ena(self, conservative_join: bool = False):
        """
        Create a paper pile to be used as a training and validation set based on ENA's related work.

        :param conservative_join: if True, look only at pairs where all annotators annotated a specific event
        :return: (df of all annotators' annotation pairs, indications of annotator identified (con,rel) pairs)
        """
        pd.set_option('display.max_colwidth', -1)

        paper_ids = ["PMC2156142", "PMC3032653", "PMC3135394", "PMC3198449", "PMC3233644", "PMC3461631", "PMC4052680",
                     "PMC4250046", "PMC4746590", "PMC534114"]

        paper_dfs = []
        annotated_connects = set()
        onto2texts, text2ontos = load_all_groundings()

        for paper_id in paper_ids:

            paper_event_dict = {}
            ena_full_text_file = f"/Users/dnsosa/Desktop/AltmanLab/bai/biotext/full_texts/{paper_id}.txt"

            # Open the full text
            with open(ena_full_text_file, 'r') as f:
                pmc_txt = f.read()

            # Create the paper
            paper = Paper(title=paper_id,
                          abstract=None,
                          full_text=re.sub('\n\n', '. ', pmc_txt),
                          doi=paper_id,
                          pmcid=paper_id,
                          pmid=None,  # DO NOT HAVE PMID FOR THESE
                          journal=None)

            annotator_events_dfs = []

            # Open the sentences
            sentences = []
            ena_sentences = f"../input/ENA_validation_data/sentences/{paper_id}_sentences.txt"
            with open(ena_sentences, 'r') as sentences_file:
                for line in sentences_file:
                    sentences.append(line.rstrip())

            for i in range(1, 4):
                # And let's check out the events
                ena_corpus_dir = "../input/ENA_validation_data/BioContext_corpus/corpus_data"
                ena_events_file = os.path.join(ena_corpus_dir, f"/annotator{i}/{paper_id}_events.tsv")
                colnames = ['event_sent_idx', 'event_span_idx', 'assoc_context_terms']
                events = pd.read_csv(ena_events_file, names=colnames, header=None, sep='\t')
                annotator_events_dfs.append(events)

            # Dropping NA because some events have no annotated context associated.
            # Conservative join means all annotators had to annotated the same relation event. Not conservative means
            # the union of individual annotators' events.
            # TODO: Check if there's a different file I should be looking at...?
            if conservative_join:
                df = reduce(lambda x, y: pd.merge(x.dropna(), y.dropna(), on='event_sent_idx', how='inner'),
                            annotator_events_dfs)[
                    ["event_sent_idx", "assoc_context_terms_x", "assoc_context_terms_y", "assoc_context_terms"]]
            else:
                df = reduce(lambda x, y: pd.merge(x.dropna(), y.dropna(), on='event_sent_idx', how='outer'),
                            annotator_events_dfs)[
                    ["event_sent_idx", "assoc_context_terms_x", "assoc_context_terms_y", "assoc_context_terms"]]
            df = df.assign(annot1_ctx_terms=df['assoc_context_terms_x'].str.split(',')).explode('annot1_ctx_terms')[:]
            df = df.reset_index().reindex(df.columns, axis=1)
            df = df.assign(annot2_ctx_terms=df['assoc_context_terms_y'].str.split(',')).explode('annot2_ctx_terms')[:]
            df = df.reset_index().reindex(df.columns, axis=1)
            df = df.assign(annot3_ctx_terms=df['assoc_context_terms'].str.split(',')).explode('annot3_ctx_terms')[:]
            df["two_agree"] = df.apply(
                lambda row: two_common(row.annot1_ctx_terms, row.annot2_ctx_terms, row.annot3_ctx_terms), 1)
            # df = df[["event_sent_idx", "annot1_ctx_terms", "annot2_ctx_terms", "annot3_ctx_terms",
            #          "two_agree"]].dropna().groupby(["event_sent_idx", "two_agree"]).apply(pd.DataFrame.sample, n=1)
            df["paper_id"] = paper_id
            paper_dfs.append(df)

            # Create the relation objects
            sent_idxs = list(set(df["event_sent_idx"]))
            sent_idxs.sort()
            for sent_idx in sent_idxs:
                relation = Relation(main_verb=None,
                                    entity1=None,
                                    entity2=None,
                                    text=sentences[sent_idx].rstrip(),
                                    paper_pmcid=paper_id,  # TODO: fix these all being the same!
                                    paper_pmid=paper_id,
                                    paper_doi=paper_id,
                                    start_idx=None,
                                    end_idx=None,
                                    sent_idx=sent_idx,
                                    sentence=Sentence(sentences[sent_idx]))

                paper.add_relation(relation)

            generic_filler_attributes = {"type": None, "pos": -9999}
            for _, row in df.iterrows():
                sent = sentences[row["event_sent_idx"]]
                for i in [1, 2, 3]:
                    term_i_id = row[f"annot{i}_ctx_terms"]
                    if term_i_id in onto2texts:
                        # for example, annotated manual:HCE, but grounding is manual:HCE-cells. Accepting the loss here.
                        term_i_text_list = onto2texts[term_i_id]
                        for term in term_i_text_list:
                            annotated_connects.add((paper_id, sent, drop_the_s(term)))
                            paper_event_dict[drop_the_s(term.lower())] = generic_filler_attributes

            # Create a context list based on the events at the intersection
            create_contexts(paper, paper_event_dict)
            self.ena_paper_pile[paper_id] = paper

        # Combine the DFs across papers
        all_annotators_df = pd.concat(paper_dfs)

        return all_annotators_df, annotated_connects

    def generate_paper_pile_from_relation_subset(self, all_res_combined, biocxml_dir, biocxml_out_dir,
                                                 con_terms_lexicon_filename):
        """
        Extract features from the pile of PMC papers intersected with PPI relations for distant supervision.

        :param all_res_combined: Pandas DataFrame with all the info needed after the query
        :param biocxml_dir: directory where biocxml files are located
        :param biocxml_out_dir: directory to write biocxml files for specific papers
        """
        all_section_names = set()
        all_subsection_names = set()
        if biocxml_out_dir is None:
            biocxml_out_dir = BIOCXML_OUT_DIR_DEFAULT

        # Convert list of document passages into one long full text string.
        def full_text_from_doc(doc):  # NOTE this could be a helper function in utilities
            full_text = ""
            for sec in doc.passages:
                # Remove the periods from et al and ig (e.g. "fig. 4") as heuristics to minimize errors from sentence
                # splitting later on
                sec_text_remove_extra_periods = sec.text.replace("et al.", "et al").replace("ig. ", "ig ")
                # The period makes section names be treated as separate sentences
                full_text += sec_text_remove_extra_periods + ". "

            # The cleaning here removes any extra '. 's that were added in above process by removing empty strings
            full_text_cleaned = list(filter(None, full_text.split(". ")))
            return '. '.join(full_text_cleaned)

        # Look for the specific filename hits in the dataframe containing the subset of relations of interest
        for filename in all_res_combined.filename.unique():

            # Get the subset of the relation subset that can be found in a specific biocxml file
            con_rel_file_df = all_res_combined[all_res_combined.filename == filename]

            # TODO: Remove this
            if not os.path.exists(os.path.join(biocxml_dir, filename)):
                print(f"skipping {os.path.join(biocxml_dir, filename)}...")
                continue

            with open(os.path.join(biocxml_dir, filename), 'rb') as f:  # check where is this

                print(f"opening {os.path.join(biocxml_dir, filename)}")
                # parser = bioc.BioCXMLDocumentReader(f)
                parser = bioc.biocxml.load(f)

                # Iterate through the documents in the biocxml collection
                for doc in parser.documents:
                    pmid = doc.infons["pmid"]
                    pmcid = doc.infons["pmcid"]
                    paper_title = doc.infons["title"]

                    # Check out the subset of relation hits there are in this specific paper.
                    con_rel_file_doc_df = con_rel_file_df[con_rel_file_df.pmcid == pmcid]  # check types

                    # If there are none (since each biocxml has >= 1 papers), we can move on and not extract features
                    if len(con_rel_file_doc_df) == 0:
                        continue
                    elif pmid not in self.pmid2contexts:
                        print(f"PMID: {pmid} not found in pmid2contexts dict.... continuing.")
                        continue
                    else:

                        # First, get the section headers information--will be useful for figuring out normalization
                        for sec in doc.passages:
                            all_section_names.add((sec.infons['section'], pmcid))
                            all_subsection_names.add((sec.infons['subsection'], pmcid))

                        pmc_text = full_text_from_doc(doc)

                        # Let's make a new paper object
                        paper = Paper(title=paper_title,
                                      abstract=None,
                                      full_text=pmc_text,
                                      doi=pmcid,  # NOTE: Using PMCID not PMID like in other places in code base
                                      pmcid=pmcid,
                                      pmid=pmid,
                                      journal=None)

                        # Get the unique relations in our subset in this paper
                        rel_list = list(con_rel_file_doc_df.rel.unique())
                        for rel in rel_list:
                            rel_reform = re.sub('<[^<]+>', "", rel)  # Remove XML tags
                            relation = Relation(main_verb=None,
                                                entity1=None,  # not worrying about it now
                                                entity2=None,
                                                text=rel_reform,  # Remove XML tags
                                                paper_pmcid=pmcid,
                                                paper_pmid=pmid,
                                                paper_doi=pmcid,  # NOTE: Using PMCID not PMID
                                                start_idx=None,
                                                end_idx=None,
                                                sent_idx=None,
                                                sentence=Sentence(rel_reform))

                            paper.add_relation(relation)

                        # Now extract context information, store it as an event_dict
                        paper_event_dict = {}
                        if con_terms_lexicon_filename is not None:
                            con_terms_df = pd.read_csv(con_terms_lexicon_filename)
                            unique_con_terms = list(set(con_terms_df.context_term))
                            # add in the plural in there
                            unique_con_terms = set(singularize_and_pluralize_list(unique_con_terms))
                            # get the subset of context terms identified in this paper
                            paper_pmid = paper.get_pmid()
                            context_hits = list(set(self.pmid2contexts[paper_pmid]).intersection(unique_con_terms))
                            # add in the plural version (simplified) to increase recall of context string matches
                            context_hits = singularize_and_pluralize_list(context_hits)
                        else:
                            context_hits = list(con_rel_file_doc_df.context_term.unique())  # TODO: double check this

                        for con in context_hits:
                            generic_filler_attributes = {"type": None, "pos": -9999}
                            paper_event_dict[con.lower()] = generic_filler_attributes

                        # This function creates all the context objects to be associated with this paper
                        create_contexts(paper, paper_event_dict)

                        # Write the BioCXML file out for each paper
                        if len(paper.get_context_list()) != 0:
                            out_file = os.path.join(biocxml_out_dir, f"{paper.get_pmcid()}.biocxml")
                            if not os.path.isfile(out_file):
                                writer = bioc.biocxml.BioCXMLDocumentWriter(out_file)
                                writer.write_collection_info(parser)
                                writer.write_document(doc)
                                writer.close()

                            # Add it to the pile
                            # Note: not doing lemmatization, just looking up direct matches
                            self.pubmed_ppi_paper_pile[pmcid] = paper  # NOTE PMCID not PMID

        return all_section_names, all_subsection_names

    def extract_features_from_all_pubmed_ppis(self, all_res_combined, biocxml_dir: str = BIOCXML_DIR_DEFAULT,
                                              biocxml_out_dir: str = BIOCXML_OUT_DIR_DEFAULT,
                                              context_type: str = None,
                                              lexicon_filename: str = None):
        """
        Extract features from the pile of PMC papers intersected with PPI relations for distant supervision.

        :param all_res_combined: Pandas DataFrame with all the info needed after the query
        :param biocxml_dir: directory where biocxml files are located
        :param biocxml_out_dir: directory where written biocxml files for specific papers are located
        :param context_type: type of context term to extract features for. Permissible: {"CTs", "tissues"}
        :param lexicon_filename: filename of lexicon of context terms we're filtering in
        """
        if biocxml_dir is None:
            biocxml_dir = BIOCXML_DIR_DEFAULT

        if biocxml_out_dir is None:
            biocxml_out_dir = BIOCXML_OUT_DIR_DEFAULT

        if lexicon_filename is not None:
            con_terms_lexicon_filename = lexicon_filename
        else:
            if context_type == "CTs":
                con_terms_lexicon_filename = CT_TERMS_LEXICON_PATH
            elif context_type == "tissues":
                con_terms_lexicon_filename = TISSUE_TERMS_LEXICON_PATH
            elif context_type == "combined":
                con_terms_lexicon_filename = CON_TERMS_LEXICON_PATH
            else:
                print("No lexicon file input. No preference for CTs and tissues. Using both together")
                con_terms_lexicon_filename = CON_TERMS_LEXICON_PATH

        # First create the paper pile
        _, _ = self.generate_paper_pile_from_relation_subset(all_res_combined, biocxml_dir, biocxml_out_dir,
                                                             con_terms_lexicon_filename)

        # Finally, extract features from the paper!
        features_df = extract_features(self.pubmed_ppi_paper_pile,
                                       do_calculate_in_mesh=True,
                                       mesh_headings_in_pmc=True,  # The PMC files now have mesh headings
                                       do_calculate_pmi=False,
                                       biocxmls_pmc_dir=biocxml_out_dir,
                                       biocxmls_pubmed_dir=biocxml_dir)

        self.pubmed_ppi_features_df = features_df
        return self.pubmed_ppi_features_df

    def output_section_names(self, all_res_combined, section_names_out_dir: str,
                             biocxml_dir: str = BIOCXML_DIR_DEFAULT,
                             biocxml_out_dir: str = BIOCXML_OUT_DIR_DEFAULT,
                             con_terms_lexicon_filename: str = CON_TERMS_LEXICON_PATH):
        """
        Extract features from the pile of PMC papers intersected with PPI relations for distant supervision.

        :param all_res_combined: Pandas DataFrame with all the info needed after the query
        :param section_names_out_dir: directory to send files containing section and subsection names
        :param biocxml_dir: directory where biocxml files are located
        :param biocxml_out_dir: directory where written biocxml files for specific papers are located
        :param con_terms_lexicon_filename: filename of lexicon of context terms we're filtering in
        """
        if biocxml_dir is None:
            biocxml_dir = BIOCXML_DIR_DEFAULT

        if biocxml_out_dir is None:
            biocxml_out_dir = BIOCXML_OUT_DIR_DEFAULT

        # First create the paper pile. The two returns are sets of tuples of section names and PMCIDs
        all_section_names, all_subsection_names = self.\
            generate_paper_pile_from_relation_subset(all_res_combined, biocxml_dir, biocxml_out_dir,
                                                     con_terms_lexicon_filename)
        section_names_df = pd.DataFrame(list(all_section_names), columns=['Section Name', 'PMCID'])
        section_names_file = os.path.join(section_names_out_dir, "insider_section_names.tsv")
        section_names_df.to_csv(section_names_file, index=False, sep='\t')

        subsection_names_df = pd.DataFrame(list(all_subsection_names), columns=['Subsection Name', 'PMCID'])
        subsection_names_file = os.path.join(section_names_out_dir, "insider_subsection_names.tsv")
        subsection_names_df.to_csv(subsection_names_file, index=False, sep='\t')

        return None

    def create_insider_corpus(self, lexicon_filename: str = None, context_type: str = None,
                              min_contexts: int = 2, max_len_token: int = 4, paper_list_outfile: str = None):
        """
        Create the insider corpus based on the extracted relations and an input lexicon to search for.

        :param lexicon_filename: filename of lexicon of context terms we're filtering in
        :param context_type: type of context term to extract features for. Permissible: {"CTs", "tissues"}
        :param min_contexts: minimum number of unique contexts from the query lexicon that must occur in each paper
        :param max_len_token: longest n-gram to search for after splitting on whitespace
        :param paper_list_outfile: file path to write list of PMCIDs for looking for features later
        """
        if lexicon_filename is not None:
            con_terms_lexicon_filename = lexicon_filename
        else:
            if context_type == "CTs":
                con_terms_lexicon_filename = CT_TERMS_LEXICON_PATH
            elif context_type == "tissues":
                con_terms_lexicon_filename = TISSUE_TERMS_LEXICON_PATH
            else:
                print("No lexicon file input. No preference for CTs and tissues. Using both together")
                con_terms_lexicon_filename = CON_TERMS_LEXICON_PATH

        con_terms_df = pd.read_csv(con_terms_lexicon_filename)
        context_terms_list = list(set(con_terms_df.context_term))

        # Get the context term set (e.g. Tabula) hits for a PMID where at least k occur
        def get_tabula_contexts(pmid, k=min_contexts):
            if pmid not in self.pmid2contexts:
                print(f"PMID: {pmid} not found in dict!")
                return "PMID not found!"

            pmid_cons_tabula = [drop_the_s(con) for con in self.pmid2contexts[pmid] if con in context_terms_list]

            if len(pmid_cons_tabula) < k:
                return "Too few contexts"
            return pmid_cons_tabula

        tot_paper_hits = 0
        context_insider_sentences_df_list = []
        for context in context_terms_list:
            # Don't query for super long contexts
            if len(context.split(' ')) > max_len_token:
                continue

            # Retrieve those relations with "in [context]" in them--these are the insider sentences
            # remove tags on protein/gene entities... shouldn't make a difference
            self.all_pubmed_relations_df['rel'] = self.all_pubmed_relations_df['rel'].str.replace('<[^<]+>', "",
                                                                                                  regex=True)
            context_insider_sentences_df_res = self.all_pubmed_relations_df[
                self.all_pubmed_relations_df['rel'].str.lower().str.contains(f" in {context}")]

            if len(context_insider_sentences_df_res) != 0:
                # Return the context hit and the other contexts in the query list found in this paper
                context_insider_sentences_df_res.loc[:, "con"] = context
                context_insider_sentences_df_res.loc[:, "context_list"] = context_insider_sentences_df_res['pmid'].\
                    apply(get_tabula_contexts)

                # How many unique papers contain an insider sentence
                num_paper_hits = len(set(context_insider_sentences_df_res.pmcid))
                if num_paper_hits > 0:
                    print(f"Number of unique papers with inside sentences --  {context}: {num_paper_hits}")
                tot_paper_hits += num_paper_hits

                context_insider_sentences_df_list.append(context_insider_sentences_df_res)

        print(f"\nTOTAL PAPER HITS COUNT: {tot_paper_hits}")

        context_insider_sentences_df = pd.concat(context_insider_sentences_df_list, axis=0)
        context_insider_sentences_df = context_insider_sentences_df.astype({'context_list': 'string'})

        # Cleaning: First, drop duplicates and instances with no contexts found
        clean_insider_df = context_insider_sentences_df[
            ['rel', 'con', 'context_list', 'pmcid', 'entity1_text', 'entity2_text',
             'distance_to_nearest_sentence_with_citation']][
            context_insider_sentences_df['context_list'] != '[]'].drop_duplicates().reset_index(drop=True)
        clean_insider_df.columns = ['Extracted Relation', 'Context', 'All Tabula Contexts in Paper', 'PMCID',
                                    'Entity 1 Text', 'Entity 2 Text', 'Citation Dist']
        n_insider_rels = len(set(clean_insider_df['Extracted Relation']))
        n_insider_papers = len(set(clean_insider_df['PMCID']))
        print(f"Total # insider sentences found in {context_type} (all) -- no context filter: {n_insider_rels}")
        print(f"Total # insider papers found in {context_type} (all) -- no context filter: {n_insider_papers}")

        # Second, drop cases where there were too few contexts
        clean_insider_df = clean_insider_df[clean_insider_df['All Tabula Contexts in Paper'] != 'Too few contexts']
        n_insider_papers = len(set(clean_insider_df['PMCID']))
        print(f"Total # insider papers found in {context_type} (all) -- WITH context filter: {n_insider_papers}")

        # Third, drop cases where the PMID wasn't found in the PMID2Context dictionary for some reason
        # TODO: Why does this happen?
        clean_insider_df = clean_insider_df[clean_insider_df['All Tabula Contexts in Paper'] != 'PMID not found!']
        n_insider_rels = len(set(clean_insider_df['Extracted Relation']))
        print(f"Total # insider sentences found in {context_type} (all): {n_insider_rels}")

        # Finally, drop cases where the citation is in insider sentence--dealing with citations is future work!
        clean_insider_df = clean_insider_df[clean_insider_df['Citation Dist'] != 0].sample(frac=1)
        clean_insider_df = clean_insider_df.reset_index(drop=True)
        n_insider_rels_nc = len(set(clean_insider_df['Extracted Relation']))
        print(f"Total # insider sentences found in {context_type} (non-citing): {n_insider_rels_nc}")

        # Get the papers for querying later
        if paper_list_outfile is not None:
            clean_insider_df.PMCID.drop_duplicates().to_csv(paper_list_outfile, index=False)

        return clean_insider_df

    def get_relation_subset_from_pmc_intersect(self, subset_df):
        """
        Return the subset of relations based on PMCs in the subset DF.

        :param subset_df: Pandas DF representing the subset corpus to be filtering on (e.g. Insider)
        """
        subset_pmc_df = subset_df.PMCID.drop_duplicates()
        relation_subset = self.all_pubmed_relations_df.\
            merge(subset_pmc_df, left_on="pmcid", right_on="PMCID", how="inner")
        n_papers = len(relation_subset.pmcid.drop_duplicates())
        print(f"After taking subset of relations, need to extract features from: {n_papers} papers")
        return relation_subset

    def get_relation_subset_from_ppi_intersect(self, subset_df):
        """
        Return the subset of relations based on PPI combined_keys in the subset DF.

        :param subset_df: Pandas DF representing the subset corpus to be filtering on (e.g. Dengue)
        """
        subset_ppis_df = subset_df.combined_entities_key.drop_duplicates()
        relation_subset = self.all_pubmed_relations_df.\
            merge(subset_ppis_df, on="combined_entities_key", how="inner")
        n_papers = len(relation_subset.pmcid.drop_duplicates())
        print(f"After taking subset of relations, need to extract features from: {n_papers} papers")
        return relation_subset
