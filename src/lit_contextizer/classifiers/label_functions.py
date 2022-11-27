"""Weak labeling functions."""

# -*- coding: utf-8 -*-

import snorkel
from textblob import TextBlob
from snorkel.labeling import labeling_function, LabelingFunction, PandasLFApplier
from snorkel.preprocess.nlp import SpacyPreprocessor
from snorkel.preprocess import preprocessor

ABSTAIN = -1
ASSOC = 1
NOT_ASSOC = 0


@labeling_function()
def num_mentions_10(x):
    return ASSOC if x.NumMentions >= 10 else ABSTAIN


@labeling_function()
def sentence_dist_10(x):
    return ASSOC if x.SentDist <= 10 else ABSTAIN


@labeling_function()
def we_our(x):
    return ASSOC if re.search(r"we|our", x.ContextSent, flags=re.I) else ABSTAIN


@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.ContextSent)
    x.polarity = scores.sentiment.polarity
    x.subjectivity = scores.sentiment.subjectivity
    return x


@labeling_function(pre=[textblob_sentiment])
def textblob_subjectivity(x):
    return ASSOC if x.subjectivity > 0.9 else ABSTAIN


def keyword_lookup(x, keywords, label):
    if any(word in x.ContextSent.lower() for word in keywords):
        return label
    return ABSTAIN


def make_keyword_lf(keywords, label):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )


@labeling_function(pre=[SpacyPreprocessor(text_field="ContextSent", doc_field="doc", memoize=True)])
def has_person(x):
    """If people are mentioned in context sentence, it's just background and maybe not relevant."""
    if any([ent.label_ == "PERSON" for ent in x.doc.ents]):
        return NOT_ASSOC
    return ABSTAIN
