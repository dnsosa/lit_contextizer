import argparse
import bioc
import os
import sys

def main():
	parser = argparse.ArgumentParser(description='Filter for documents with desired dependency paths')
	parser.add_argument('--inBioc',required=True,type=str,help='Input BioC file with relations')
	parser.add_argument('--annotations',required=True,type=str,help='Annotations of dependency paths')
	parser.add_argument('--entityTypes',required=True,type=str,help='Comma-delimited pair of entity types (e.g. Gene,Gene)')
	parser.add_argument('--outBioc',required=True,type=str,help='Output file')
	args = parser.parse_args()

	wanted_type1,wanted_type2 = args.entityTypes.split(',')
	
	wanted_deppaths = set()
	with open(args.annotations) as f:
		for line in f:
			anno,deppath,sentence = line.rstrip('\n').split('\t')
			if not anno in ['','x','?']:
				wanted_deppaths.add(deppath)

	print("Filtering...")

	writer = bioc.BioCXMLDocumentWriter(args.outBioc)
	with open(args.inBioc,'rb') as inF:
		parser = bioc.BioCXMLDocumentReader(inF)
		
		for doc in parser:
			if not 'pmid' in doc.infons or doc.infons['pmid'] in ['','None']:
				continue

			for p in doc.passages:
				entitytypes = { a.id:a.infons['type'] for a in p.annotations }

				filtered_relations = []
				for r in p.relations:
					t1 = entitytypes[r.nodes[0].refid]
					t2 = entitytypes[r.nodes[1].refid]
					deppath = r.infons['deppath']
					if t1 == wanted_type1 and t2 == wanted_type2 and deppath in wanted_deppaths:
					#if deppath in wanted_deppaths:
						filtered_relations.append(r)

				p.relations = filtered_relations

			relation_count = len( [ r for p in doc.passages for r in p.relations ] )
			if relation_count > 0:
				writer.write_document(doc)

	writer.close()
	print("Done")


if __name__ == '__main__':
	main()

