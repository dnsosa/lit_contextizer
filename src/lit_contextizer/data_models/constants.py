# -*- coding: utf-8 -*-

"""Constants for data_models in lit_contextizer."""

import os

HERE = os.path.abspath(os.path.dirname(__file__))
RESOURCES = os.path.join(HERE, 'resources')

#CON_TERMS_LEXICON_PATH = None
CON_TERMS_LEXICON_PATH = os.path.join(RESOURCES, 'tabula_tissues_CTs.csv')
CT_TERMS_LEXICON_PATH = os.path.join(RESOURCES, 'tabula_CTs.csv')
TISSUE_TERMS_LEXICON_PATH = os.path.join(RESOURCES, 'tabula_tissues.csv')
BACKGROUND_SYNS_PATH = os.path.join(RESOURCES, 'background_synonyms.csv')
METHODS_SYNS_PATH = os.path.join(RESOURCES, 'methods_synonyms.csv')
RESULTS_SYNS_PATH = os.path.join(RESOURCES, 'results_synonyms.csv')
DISCCONC_SYNS_PATH = os.path.join(RESOURCES, 'disc_conc_synonyms.csv')
DENGUE_PPIS_PATH = os.path.join(RESOURCES, 'dengue_target-target_paras.csv')

