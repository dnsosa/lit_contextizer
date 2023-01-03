# -*- coding: utf-8 -*-

"""Constants for data_models in lit_contextizer."""

import os

HERE = os.path.abspath(os.path.dirname(__file__))
RESOURCES = os.path.join(HERE, 'resources')

CON_TERMS_LEXICON_PATH = os.path.join(RESOURCES, 'tabula_tissues_CTs.csv')
CT_TERMS_LEXICON_PATH = os.path.join(RESOURCES, 'tabula_CTs.csv')
TISSUE_TERMS_LEXICON_PATH = os.path.join(RESOURCES, 'tabula_tissues.csv')
BACKGROUND_SYNS_PATH = os.path.join(RESOURCES, 'background_synonyms.csv')
METHODS_SYNS_PATH = os.path.join(RESOURCES, 'methods_synonyms.csv')
RESULTS_SYNS_PATH = os.path.join(RESOURCES, 'results_synonyms.csv')
DISCCONC_SYNS_PATH = os.path.join(RESOURCES, 'disc_conc_synonyms.csv')
DENGUE_PPIS_PATH = os.path.join(RESOURCES, 'dengue_ppi_paras.csv')

LOCAL_HOME = "/Users/dnsosa/Desktop/AltmanLab/"
ENA_VAL_DIR = os.path.join(LOCAL_HOME, "bai/Stanford-Collab/input/ENA_validation_data")
ALL_GROUNDINGS_PATH = os.path.join(ENA_VAL_DIR, "/BioContext_corpus/corpus_data/all_groundings.tsv")

RELATIONSHIPS_FILE = "../input/opensource_gene_gene_sign_contradictions_svo_w_license_strict_w_entity_locations.csv"
FULL_TEXTS_FILE = "../input/annotated_documents.json"
SECTION_MAPPER_MASTER_FILE = "/Users/dnsosa/Desktop/AltmanLab/bai/biotext/full_texts/dan_query_docs_pmids.bioc.xml"

CLUSTER_GRP = "/oak/stanford/groups/rbaltman/"
BIOCXML_DIR_WITH_CITATION_DISTS = "jlever/bai-stanford-collab/relation_extraction/working/with_tidy_citation_distances"
BIOCXML_DIR_DEFAULT = os.path.join(CLUSTER_GRP, BIOCXML_DIR_WITH_CITATION_DISTS)
BIOCXML_OUT_DIR_DEFAULT = os.path.join(CLUSTER_GRP, "dnsosa/bai_lit_context/Stanford-Collab/biocxml_out")
