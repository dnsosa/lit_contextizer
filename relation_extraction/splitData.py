import argparse
import os
import sys

def main():
	parser = argparse.ArgumentParser(description='Split the GNBR2 data into separate files and remove duplicate documents')
	parser.add_argument('--inDir',required=True,type=str,help='Input directory with dependency path data')
	parser.add_argument('--outDir',required=True,type=str,help='Output directory')
	args = parser.parse_args()

	pubmed_input_files = [ f for f in os.listdir(args.inDir) if f.startswith('pubmed') and f.endswith('.tsv') ]
	pmc_input_files = [ f for f in os.listdir(args.inDir) if f.startswith('pmc') and f.endswith('.tsv') ]

	input_files = sorted(pmc_input_files,reverse=True) + sorted(pubmed_input_files,reverse=True)

	seen_pmids = set()
	out_handles = {}

	for input_file in input_files:
		print("Processing %s" % input_file)
		sys.stdout.flush()
		
		pmids_in_file = set()
		with open(os.path.join(args.inDir,input_file)) as f:
			for line in f:
				split = line.strip('\n').split('\t')

				pmid, type1, id1, type2, id2, path_length, path, sentence = split
				
				pmid = int(pmid)
				if pmid in seen_pmids:
					continue
				pmids_in_file.add(pmid)
				
				out_file_key = (type1,type2)
				if not out_file_key in out_handles:
					out_handles[out_file_key] = open(os.path.join(args.outDir,"%s_%s.tsv" % out_file_key),'w')
				out_handle = out_handles[out_file_key]
				
				out_data = [ str(pmid), id1, id2, path_length, path, sentence ]
				out_handle.write("\t".join(out_data) + "\n" )

		seen_pmids.update(pmids_in_file)

if __name__ == '__main__':
	main()
