import argparse
import spacy
import bioc
from intervaltree import IntervalTree

def main():
    parser = argparse.ArgumentParser(description='Parses a BioC file containing relations and citations and calculates the distance between each relation and citation')
    parser.add_argument('--inBioc',required=True,type=str,help='Input file in BioC format')
    parser.add_argument('--outBioc',required=True,type=str,help='Output file in BioC format')
    args = parser.parse_args()
    
    nlp = spacy.load("en_core_sci_sm")
    
    # Load input file, iterate through each document & passage, process and save
    writer = bioc.biocxml.BioCXMLDocumentWriter(args.outBioc)
    with open(args.inBioc,'rb') as f:
        parser = bioc.biocxml.BioCXMLDocumentReader(f)
        for doc in parser:            
            for passage in doc.passages:
                # Skip processing passages that don't have relations
                if len(passage.relations) == 0:
                    continue
                    
                passage_end = passage.offset + len(passage.text)
                
                parsed = nlp(passage.text)
                
                # Get the starting positions of each sentence
                sentence_starts = []
                for sentence in parsed.sents:
                    sentence_start = passage.offset + sentence[0].idx
                    sentence_starts.append(sentence_start)
                    
                # Create an IntervalTree of the sentence locations for easy sentence index lookup
                sentence_tree = IntervalTree()
                for i,sentence_start in enumerate(sentence_starts):
                    sentence_end = sentence_starts[i+1] if (i+1) < len(sentence_starts) else passage_end
                    
                    sentence_tree[sentence_start:sentence_end] = i
                
                # Figure out which sentences contain citations (and save their indices)
                sentences_with_citations = set()
                for a in passage.annotations:
                    # Filter for citations (that either have bibr as a ref-type or provide a PMID)
                    if a.infons['type'] == 'xref' and (a.infons.get('ref-type') == 'bibr' or a.infons.get('pmid')):
                        ref_start = a.locations[0].offset
                        sentence_no = list(sentence_tree[ref_start])[0].data
                        sentences_with_citations.add(sentence_no)
                sentences_with_citations = sorted(sentences_with_citations)
                
                
                if len(sentences_with_citations) == 0:
                    # There are no sentences with citations so save NA to each relation
                    for r in passage.relations:
                        r.infons['distance_to_nearest_sentence_with_citation'] = 'NA'
                else:
                    # Figure out which sentence each relation is in, find the closest sentence with a citation, and save into the relation
                    for r in passage.relations:
                        sentence_start = int(r.infons['sentence_start'])
                        sentence_no = list(sentence_tree[sentence_start])[0].data
                        
                        min_distance = min( [ abs(x-sentence_no) for x in sentences_with_citations ] )
                        
                        r.infons['distance_to_nearest_sentence_with_citation'] = min_distance
                
                        
            writer.write_document(doc)
            
    print("Done")

if __name__ == '__main__':
    main()
