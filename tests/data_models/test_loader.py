"""Tests for loading the data and creating the appropriate objects."""

# -*- coding: utf-8 -*-

import unittest

from lit_contextizer.data_models.DataLoader import DataLoader
from lit_contextizer.data_models.Paper import Paper


# TODO: maybe rename these tests so less confusing
class TestLoadDataset(unittest.TestCase):
    """Tests loading of data and creating objects."""

    def setUp(self) -> None:
        """Set up."""
        self.loader = DataLoader()

    def test_parse_annotated_texts(self):
        """Test parsing annotated texts."""
        # TODO: Test for fixed XML
        # TODO: Test for abstract vs full text distinction
        # TODO: Test sentence index vs. found index in Paper class
        # TODO: Test that paper is updated after new context is found

        annotated_text_test_file = "tests/test_inputs/annots_test_file.json"
        self.loader.parse_annotated_full_texts(in_file=annotated_text_test_file)

        # Make a better test file
        # self.assertEquals(len(self.loader.events), 4)
        self.assertEqual(len(self.loader.paper_pile), 3)
        self.assertTrue("FakePaperDOI1" in self.loader.paper_pile)
        self.assertEqual(type(self.loader.paper_pile["FakePaperDOI1"]), Paper)

    def test_parse_relationships(self):
        """Test parsing SVO PPI relationships."""
        # TODO: Test things are being properly added (count, look for specific instance)
        # TODO: Test that the paper_pile is updated if new paper is found
        # TODO: Test that paper is updated with new relationships after new relationship is found
        # TODO: Test that I coan find the distance between a relation sentence and a context sentence in same paper

        relationships_test_file = "tests/test_inputs/relationships_test_file.csv"
        self.loader.parse_relationships_file(relationships_test_file)

        self.assertEqual(len(self.loader.relationships), 2)
        # TODO: Note I was planning to create a version of this test co-dependent on the previous test to ensure update
        self.assertEqual(len(self.loader.paper_pile), 2)  # added one new paper here if dependent on previous test
        self.assertEqual(len(self.loader.paper_pile["FakePaperDOI1"].get_relations()), 2)
