"""Tests for computation using the paper object."""

# -*- coding: utf-8 -*-

import unittest

from lit_contextizer.data_models.Extractable import Context, Relation
from lit_contextizer.data_models.Paper import Paper
from lit_contextizer.data_models.Sentence import Sentence


class TestMakeDataset(unittest.TestCase):
    """Tests some functionality of the paper class, which enables feature extraction."""

    def test_find_sentence_idx(self):
        """Test finding sentence ID in a paper."""
        p1 = Paper("Fake Paper", "This paper is a fake one.",
                   "This paper is fake. The contents are not real. The results show that it is still fake.")
        s1 = Sentence("This paper is totally fake.")
        s2 = Sentence("the contents aren't real")

        self.assertEqual(p1.find_sentence_idx(s1)[0], 0)
        self.assertEqual(p1.find_sentence_idx(s2)[0], 1)

    def test_extracting_features(self):
        """Test some of the features being extracted."""
        p2_full_text = "This paper is not real. We had noticed that EGFR upregulates PCSK9, which is very cool. " \
                       "These results will be very high impact. By the way this was in liver cells. " \
                       "This was not in cardiac cells. But we love cardiac cells."
        p2 = Paper("Fake Paper 2", "This paper is also fake.", p2_full_text, doi="1234")
        r1 = Relation(main_verb="upregulates",
                      text="We had noticed that EGFR upregulates PCSK9, which is very cool.",
                      paper_doi="1234",
                      start_idx=44,
                      end_idx=65,
                      sent_idx=1,
                      sentence=Sentence("We had noticed that EGFR upregulates PCSK9, which is very cool."))
        c1 = Context(text="cardiac",
                     paper_doi="1234",
                     start_idx=182,
                     end_idx=189,
                     sent_idx=4,
                     uuid="5678",
                     ctx_type="CellType",
                     ctx_uid="92837",
                     sentence=Sentence("This was not in cardiac cells."))
        c2 = Context(text="liver",
                     paper_doi="1234",
                     start_idx=152,
                     end_idx=157,
                     sent_idx=3,
                     uuid="5678",
                     ctx_type="CellType",
                     ctx_uid="10293",
                     sentence=Sentence("By the way this was in liver cells."))

        self.assertEqual(p2.sentence_distance(r1, c2), 2)
        self.assertEqual(p2.sentence_distance(r1, c1), 3)
        self.assertEqual(p2.get_tense(c2), "past")
        self.assertEqual(p2.get_tense(r1), "past")  # Inconclusive, need to decide a rule
        self.assertEqual(p2.get_pov(c2), set())
        self.assertEqual(p2.get_pov(r1), {'1'})
        self.assertEqual(p2.dp_distance(c2, r1), 7)  # Unsure, need to double check
