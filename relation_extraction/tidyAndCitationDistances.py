import argparse
import spacy
import bioc
from intervaltree import IntervalTree
import re

def stepPastNumericRefs(passage, refs, loc, clamp_right=False):
    new_loc = loc
    found_more = True
    while found_more:
        found_more = False
    
        next_refs = [ (s,e) for s,e in refs if s >= new_loc ]
        
        if next_refs:
            ref_start,ref_end = next_refs[0]
            
            # Check it's a numeric (ish) citation and not a Blah et al.
            ref_text = passage.text[ref_start:ref_end]
            if re.match(r'^[\d+\-]+$',ref_text):            
                between_text = passage.text[new_loc:ref_start]
                
                between_text = between_text.strip(' ,-')
                if between_text == '':
                    new_loc = ref_end
                    
                    if clamp_right:
                        next_text = passage.text[ref_end:]
                        extra_spaces = len(next_text) - len(next_text.lstrip())
                        new_loc += extra_spaces
                                            
                    found_more = True
                
    return new_loc
    
    

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
                    
                passage_end = len(passage.text)
                
                parsed = nlp(passage.text)
                
                # Get citation locations
                refs = []
                for a in passage.annotations:
                    # Filter for citations (that either have bibr as a ref-type or provide a PMID)
                    if a.infons['type'] == 'xref' and (a.infons.get('ref-type') == 'bibr' or a.infons.get('pmid')):
                        ref_start = a.locations[0].offset - passage.offset
                        ref_end = ref_start + a.locations[0].length
                        
                        refs.append( (ref_start, ref_end) )
                        
                refs = sorted(refs)

                # Get the starting positions of each sentence
                sentence_starts = []
                for sentence in parsed.sents:
                    sentence_start = sentence[0].idx
                    sentence_end = sentence[-1].idx + len(sentence[-1].text)
                    
                    new_sentence_start = stepPastNumericRefs(passage, refs, sentence_start, clamp_right=True)
                    #new_sentence_end = stepPastNumericRefs(passage, refs, sentence_end)
                    
                    if sentence_start != new_sentence_start:
                        in_between = passage.text[sentence_start:new_sentence_start]
                        #print(in_between)
                    
                    sentence_starts.append(new_sentence_start)
                    
                sentence_starts = sorted(set(sentence_starts))
                            
                # Create an IntervalTree of the sentence locations for easy sentence index lookup
                sentence_tree = IntervalTree()
                for i,sentence_start in enumerate(sentence_starts):
                    if sentence_start != passage_end:
                        sentence_end = sentence_starts[i+1] if (i+1) < len(sentence_starts) else passage_end
                        
                        sentence_tree[sentence_start:sentence_end] = i
                        
                # Add in an interval before the first sentence if needed
                if sentence_starts[0] > 0:
                    sentence_tree[0:sentence_starts[0]] = -1
                
                # Figure out which sentences contain citations (and save their indices)
                sentences_with_citations = set()
                for ref_start,ref_end in refs:
                    sentence_no = list(sentence_tree[ref_start])[0].data
                    sentences_with_citations.add(sentence_no)
                sentences_with_citations = sorted(sentences_with_citations)
                
                # Deal with sentence boundary modifications where there are citations that should be excluded from the start of a sentence and included at the end.
                for r in passage.relations:
                    sentence_start = int(r.infons['sentence_start']) - passage.offset
                    sentence_end = int(r.infons['sentence_end']) - passage.offset
                    
                    new_sentence_start = stepPastNumericRefs(passage, refs, sentence_start, clamp_right=True)
                    new_sentence_end = stepPastNumericRefs(passage, refs, sentence_end)
                    
                    r.infons['sentence_start'] = new_sentence_start + passage.offset
                    r.infons['sentence_end'] = new_sentence_end + passage.offset
                    
                    # Trim off the start of the sentence (for citations erroneously included)
                    if sentence_start != new_sentence_start:
                        diff = (new_sentence_start-sentence_start)
                        r.infons['formatted_sentence'] = r.infons['formatted_sentence'][diff:]
                        
                    # Add to the end of a sentence (to include relevant citations after the sentence end)
                    if sentence_end != new_sentence_end:
                        diff = (new_sentence_end-sentence_end)
                        r.infons['formatted_sentence'] = r.infons['formatted_sentence'] + passage.text[sentence_end:new_sentence_end]
                
                if len(sentences_with_citations) == 0:
                    # There are no sentences with citations so save NA to each relation
                    for r in passage.relations:
                        r.infons['distance_to_nearest_sentence_with_citation'] = 'NA'
                else:
                    # Figure out which sentence each relation is in, find the closest sentence with a citation, and save into the relation
                    for r in passage.relations:
                        sentence_start = int(r.infons['sentence_start']) - passage.offset
                        
                        sentence_no = list(sentence_tree[sentence_start])[0].data
                        
                        min_distance = min( [ abs(x-sentence_no) for x in sentences_with_citations ] )
                        
                        r.infons['distance_to_nearest_sentence_with_citation'] = min_distance
                    
                        
            writer.write_document(doc)
            
    print("Done")

if __name__ == '__main__':
    main()
