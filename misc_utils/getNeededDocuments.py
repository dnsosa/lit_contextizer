import argparse
import bioc
import sys
from collections import defaultdict
import os

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Get documents given a PMID list')
	parser.add_argument('--pmids',required=True,type=str,help='PMID list')
	#parser.add_argument('--inBioc',required=True,type=str,help='Input BioC file')
	parser.add_argument('--inDir',required=True,type=str,help='Input directory with BioC files')
	parser.add_argument('--outBioc',required=True,type=str,help='Output BioC File')
	args = parser.parse_args()

	pmids = set()

	print("Loading PMIDs...")
	sys.stdout.flush()

	with open(args.pmids) as f:
		wanted_pmids = set( int(line.strip()) for line in f )

	print("Filtering files with PMIDs...")
	sys.stdout.flush()

	pubmed_input_files = [ f for f in os.listdir(args.inDir) if f.startswith('pubmed') and f.endswith('.bioc.xml') ]
	pmc_input_files = [ f for f in os.listdir(args.inDir) if f.startswith('pmc') and f.endswith('.bioc.xml') ]

	input_files = sorted(pmc_input_files,reverse=True) + sorted(pubmed_input_files,reverse=True)

	writer = bioc.BioCXMLDocumentWriter(args.outBioc)
	for input_file in input_files:
		print("  %s - %d PMIDs remaining" % (input_file,len(wanted_pmids)))
		sys.stdout.flush()

		with open(os.path.join(args.inDir,input_file),'rb') as f:
			parser = bioc.BioCXMLDocumentReader(f)
			for i,doc in enumerate(parser):
				if not ('pmid' in doc.infons and doc.infons['pmid'] and doc.infons['pmid'] != 'None'):
					continue

				pmid = int(doc.infons['pmid'])

				if pmid in wanted_pmids:
					writer.write_document(doc)
					wanted_pmids.remove(pmid)
					
		if len(wanted_pmids) == 0:
			break

	print("Done.")
	sys.stdout.flush()

