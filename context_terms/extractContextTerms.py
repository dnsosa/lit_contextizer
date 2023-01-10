import argparse
import bioc
import pickle
from collections import defaultdict,Counter
import re
import sys
import string
import kindred
from tqdm import tqdm

def main():
	parser = argparse.ArgumentParser('Do NER to find context terms')
	parser.add_argument('--inBioc',required=True,type=str,help='Input BioC file')
	parser.add_argument('--contextsWithIDs',required=False,type=str,help='Word-list with context terms (three column including ID and synonyms)')
	parser.add_argument('--contextsWithoutIDs',required=False,type=str,help='Word-list with context terms (one column)')
	parser.add_argument('--outBioc',required=True,type=str,help='Output BioC file')
	parser.add_argument('--verbose',action='store_true',help='Whether to give more output')
	args = parser.parse_args()

	pmids = set()

	print("WARNING: No entity ambiguity is allowed. Each phrase is linked with a single context (e.g. PC3 only links to a single cell-line and not all PC3 cellines. Proper disambiguation is a needed later stage")
	print()

	assert args.contextsWithIDs or args.contextsWithoutIDs

	termLookup = defaultdict(set)
	id2TermAndType = {}

	if args.contextsWithIDs:
		with open(args.contextsWithIDs) as f:
			for line in f:
				termID,name,synonyms,termType = line.rstrip('\n').split('\t')
				for synonym in synonyms.split('|'):
					if synonym.lower() in termLookup:
						continue

					termLookup[synonym.lower()].add(('CellContext',termID))
				id2TermAndType[termID] = (name,termType)
	elif args.contextsWithoutIDs:
		with open(args.contextsWithoutIDs) as f:
			for term in f:
				term = term.lower().strip()
				if term:
					termLookup[term].add(('CellContext',term))
					termLookup[term+'s'].add(('CellContext',term)) # Check for plurals
					termLookup[term+'es'].add(('CellContext',term)) # Check for plurals
					id2TermAndType[term] = (term,'CellContext')
			
	print("Starting text alignment...")

	currentID = 1
	writer = bioc.biocxml.BioCXMLDocumentWriter(args.outBioc,encoding='UTF-8')
	with open(args.inBioc,'rb') as f:
		parser = bioc.biocxml.BioCXMLDocumentReader(f)

		iterator = tqdm(enumerate(parser)) if args.verbose else enumerate(parser)

		for i,doc in iterator:
			corpus = kindred.Corpus()
			for p in doc.passages:
				corpus.addDocument(kindred.Document(p.text))

			parser = kindred.Parser(model='en_core_sci_sm')
			parser.parse(corpus)

			ner = kindred.EntityRecognizer(termLookup, mergeTerms=True)
			ner.annotate(corpus)

			for kDoc,passage in zip(corpus.documents, doc.passages):
				existingIDs = set( a.id for a in passage.annotations )
				for entity in kDoc.entities:
					start,end = entity.position[0]
					normalized,subtype = id2TermAndType[entity.externalID]

					a = bioc.BioCAnnotation()
					a.text = passage.text[start:end]
					a.infons = {'type':entity.entityType, 'conceptid': entity.externalID, 'normalized':normalized, 'subtype':subtype}
					a.id = 'C%d' % currentID
					assert not a.id in existingIDs
					currentID += 1

					if end <= start:
						continue

					biocLoc = bioc.BioCLocation(offset=passage.offset+start, length=(end-start))
					a.locations.append(biocLoc)
					passage.annotations.append(a)

			writer.write_document(doc)

	print ('Done!')

if __name__ == '__main__':
	main()

