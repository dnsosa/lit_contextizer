"""
Class for storing extractable events from text, namely context mentions and relations.

Note that Extractables will only exist in the context of the paper that "holds" it.
"""

# -*- coding: utf-8 -*-

import pandas as pd

from biothings_client import get_client
from lit_contextizer.data_models.constants import DENGUE_PPIS_PATH


def create_dengue_corpus(dengue_filename: str = DENGUE_PPIS_PATH):
    """Create the dengue corpus (mostly reformatting)."""
    dengue_df = pd.read_csv(dengue_filename, index_col=[0])

    dengue_pps_df = dengue_df[["name1", "name2"]]
    dengue_pps_df.columns = ["Gene1", "Gene2"]
    print(f"Dengue_pps_df network has: {len(dengue_pps_df.drop_duplicates())} directed edges")

    all_genes_in_net = set(dengue_pps_df["Gene1"].values).union(set(dengue_pps_df["Gene2"].values))
    print(f"There are: {len(all_genes_in_net)} unique genes in this network")

    mg = get_client('gene')

    gene_map_query = mg.querymany(all_genes_in_net, scopes="symbol", fields="entrezgene", species="human")
    gene_dict = {gene["query"]: int(gene["entrezgene"]) for gene in gene_map_query if "entrezgene" in gene.keys()}

    print("Retrieved a mapping from Entrez to gene symbols")

    dengue_pps_df["Gene1_Entrez"] = dengue_pps_df["Gene1"].map(gene_dict)
    dengue_pps_df["Gene2_Entrez"] = dengue_pps_df["Gene2"].map(gene_dict)

    dengue_pps_df = dengue_pps_df.dropna()

    dengue_pps_df["Gene1_Entrez"] = dengue_pps_df["Gene1_Entrez"].astype('int')
    dengue_pps_df["Gene2_Entrez"] = dengue_pps_df["Gene2_Entrez"].astype('int')

    def make_keys3(x):
        sorted_entrez = sorted(
            [str(x['Gene1_Entrez']), str(x['Gene2_Entrez'])])  # Alphabetically NOT numerically sorted
        return '_'.join(sorted_entrez)

    dengue_pps_df['combined_entities_key'] = dengue_pps_df.apply(make_keys3, axis=1)

    return dengue_pps_df
