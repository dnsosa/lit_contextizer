import argparse
import bioc
import os
import sys
import random

def main():
	parser = argparse.ArgumentParser(description='Select a random sample of documents containing dependency paths of interest')
	parser.add_argument('--inDir',required=True,type=str,help='Input directory of BioC file with relations')
	parser.add_argument('--sampleSize',required=True,type=int,help='Number of documents to randomly select')
	parser.add_argument('--outFile',required=True,type=str,help='Output file')
	args = parser.parse_args()

	pubmed_files = [ f for f in os.listdir(args.inDir) if f.startswith('pubmed') and f.endswith('.bioc.xml') ]
	pmc_files = [ f for f in os.listdir(args.inDir) if f.startswith('pmc') and f.endswith('.bioc.xml') ]

	input_files = sorted(pmc_files,reverse=True) + sorted(pubmed_files,reverse=True)

	seen_pmids = {}

	print("Finding PMIDs")
	for input_file in input_files:
		with open(os.path.join(args.inDir,input_file),'rb') as inF:
			print("  Processing %s" % input_file)
			sys.stdout.flush()

			parser = bioc.biocxml.BioCXMLDocumentReader(inF)
			
			for doc in parser:
				if not 'pmid' in doc.infons or doc.infons['pmid'] in ['','None']:
					continue

				pmid = int(doc.infons['pmid'])
				if pmid in seen_pmids:
					continue

				seen_pmids[pmid] = input_file

	chosen_pmids = random.sample(seen_pmids.keys(), args.sampleSize)

	print("Randomly selected %d PMIDs from %d documents for sample" % (len(chosen_pmids),len(seen_pmids)))
	print()

	chosen_files = set( seen_pmids[pmid] for pmid in chosen_pmids )
	chosen_files_with_nice_order = [ input_file for input_file in input_files if input_file in chosen_files ]

	print("Fetching documents with sampled PMID list from %d files" % len(chosen_files_with_nice_order))

	writer = bioc.biocxml.BioCXMLDocumentWriter(args.outFile)
	for input_file in chosen_files_with_nice_order:
		if len(chosen_pmids) == 0:
			break

		with open(os.path.join(args.inDir,input_file),'rb') as inF:
			print("  Processing %s. Got %d to find" % (input_file,len(chosen_pmids)))
			sys.stdout.flush()

			parser = bioc.biocxml.BioCXMLDocumentReader(inF)
			
			for doc in parser:
				if not 'pmid' in doc.infons or doc.infons['pmid'] in ['','None']:
					continue

				pmid = int(doc.infons['pmid'])
				if not pmid in chosen_pmids:
					continue
				chosen_pmids.remove(pmid)

				writer.write_document(doc)

	writer.close()
	print("Done")


if __name__ == '__main__':
	main()

