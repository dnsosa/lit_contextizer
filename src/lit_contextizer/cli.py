"""Command line interface for lit-contextizer."""

# -*- coding: utf-8 -*-

import json
import os

import click

import pandas as pd

from .data_models.DataLoader import DataLoader
from .data_models.DataLoaderUtilities import create_dengue_corpus


@click.command()
@click.option('--output_folder', 'out_dir')
@click.option('--networks_folder', 'networks_dir')
@click.option('--full_text_dir', 'full_text_dir', default=None)
@click.option('--paper_subset', 'paper_subset', default="insider")
@click.option('--insider_context_type', 'insider_context_type', default=None)
@click.option('--parse_files/--no_parse_files', 'parse_files', default=True)
@click.option('--dump_annots_context/--no_dump_annots_context', 'dump_annots_context', default=False)
@click.option('--load_max', 'load_max', default=float("inf"))
@click.option('--min_conf', 'min_conf', default=0.0)
def main(out_dir, networks_dir, full_text_dir, paper_subset, insider_context_type, parse_files, dump_annots_context,
         load_max, min_conf):
    """Run main function."""
    dl = DataLoader()
    print("Done initializing")

    if parse_files:
        dl.parse_pubmed_full_texts(load_max=load_max, in_dir=full_text_dir)
        print("Done with full texts")
        all_relations_file = os.path.join(out_dir, "all_pubmed_relations_citationDist_v2.tsv")
        dl.all_pubmed_relations_df.to_csv(all_relations_file, sep='\t', index=False)

        def set_default(obj):
            if isinstance(obj, set):
                return list(obj)
            raise TypeError

        if dump_annots_context:
            with open(os.path.join(out_dir, "annot_id2entrez.csv"), 'w') as fp:
                json.dump(dl.annot_id2entrez, fp, default=set_default)
            with open(os.path.join(out_dir, "annot_id2text.csv"), 'w') as fp:
                json.dump(dl.annot_id2text, fp, default=set_default)
            with open(os.path.join(out_dir, "pmid2contexts.csv"), 'w') as fp:
                json.dump(dl.pmid2contexts, fp, default=set_default)
            with open(os.path.join(out_dir, "contexts2pmid.csv"), 'w') as fp:
                json.dump(dl.contexts2pmid, fp, default=set_default)
            with open(os.path.join(out_dir, "pmid2species.csv"), 'w') as fp:
                json.dump(dl.pmid2species, fp, default=set_default)

    else:
        all_relations_file = os.path.join(out_dir, "all_pubmed_relations_citationDist_v2.tsv")
        dl.all_pubmed_relations_df = pd.read_csv(all_relations_file, sep='\t').astype({'pmid': 'string'})
        dl.annot_id2entrez = json.load(open(f'{out_dir}/annot_id2entrez.csv'))
        dl.annot_id2text = json.load(open(f'{out_dir}/annot_id2text.csv'))
        dl.pmid2contexts = json.load(open(f'{out_dir}/pmid2contexts.csv'))
        dl.contexts2pmid = json.load(open(f'{out_dir}/contexts2pmid.csv'))
        dl.pmid2species = json.load(open(f'{out_dir}/pmid2species.csv'))

    if paper_subset == "insider":
        print(f"Creating insider corpus for insider context type: {insider_context_type}")
        insider_df = dl.create_insider_corpus(context_type=insider_context_type)
        relation_subset = dl.get_relation_subset_from_pmc_intersect(subset_df=insider_df)
        # dl.output_section_names(relation_subset, out_dir)  # output section names for normalizing
        if full_text_dir is not None:
            biocxml_out_dir = os.path.join(full_text_dir, "../biocxml_out")
        else:
            biocxml_out_dir = None
        features_df = dl.extract_features_from_all_pubmed_ppis(relation_subset,
                                                               biocxml_dir=full_text_dir,
                                                               biocxml_out_dir=biocxml_out_dir,
                                                               context_type=insider_context_type)

        context_type_filename = 'combined' if insider_context_type is None else insider_context_type
        insider_full_filename = os.path.join(out_dir, f"{context_type_filename}_insider_papers_features_df.tsv")
        features_df.to_csv(insider_full_filename, index=False, sep='\t')

    elif paper_subset == "dengue":
        print("Finding the subset of relations from my input Dengue pairs I need to extract features about.")
        dengue_ppis_df = create_dengue_corpus()
        print("PPI Dengue corpus created.")
        relation_subset = dl.get_relation_subset_from_ppi_intersect(subset_df=dengue_ppis_df)
        print("Got subset of Dengue relations")
        # dl.output_section_names(relation_subset, out_dir)  # output section names for normalizing
        features_df = dl.extract_features_from_all_pubmed_ppis(relation_subset, context_type="CTs")
        features_df.to_csv(os.path.join(out_dir, "dengue_papers_features_df.tsv"), index=False, sep='\t')

    elif paper_subset == "giant":
        context_list = ['adipose_tissue', 'liver', 'lung']
        context_lookup_list = ['adipo', 'liver', 'lung']

        for context, context_lookup in zip(context_list, context_lookup_list):
            topnet_file = os.path.join(networks_dir, f"{context}_top")
            context_query_res_df = dl.query_context_specific_ppis([context_lookup], topnet_file, min_conf=min_conf)
            print(f"Found {len(context_query_res_df)} results from intersecting with the {context} PPI")
            context_query_res_df.to_csv(os.path.join(out_dir, f"{context}_ppi_pmc_df_conf{min_conf}.csv"), index=False)

        adipose_res_path = os.path.join(out_dir, f"adipose_tissue_ppi_pmc_df_conf{min_conf}.csv")
        liver_res_path = os.path.join(out_dir, f"liver_ppi_pmc_df_conf{min_conf}.csv")
        lung_res_path = os.path.join(out_dir, f"lung_ppi_pmc_df_conf{min_conf}.csv")

        adipose_res = pd.read_csv(adipose_res_path)
        liver_res = pd.read_csv(liver_res_path)
        lung_res = pd.read_csv(lung_res_path)

        adipose_res["context_term"] = "adipose"
        adipose_res["context_term_prefix"] = "adipo"
        liver_res["context_term"] = "liver"
        liver_res["context_term_prefix"] = "liver"
        lung_res["context_term"] = "lung"
        lung_res["context_term_prefix"] = "lung"

        all_res_combined = pd.concat([adipose_res, liver_res, lung_res], ignore_index=True)
        features_df = dl.extract_features_from_all_pubmed_ppis(all_res_combined, context_type="tissues")
        features_df.to_csv(os.path.join(out_dir, f"ppi_pmc_tissues_features_df_conf{min_conf}.csv"), index=False)


if __name__ == '__main__':
    main()
