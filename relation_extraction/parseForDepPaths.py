import argparse
import spacy
import bioc
from intervaltree import IntervalTree
from collections import defaultdict
import networkx as nx
import itertools
from networkx.algorithms.shortest_paths.generic import shortest_path
import html
import re
import time

nlp = spacy.load("en_core_sci_sm")
def sentencesGenerator(text):
    parsed = nlp(text)
    sentence = None
    for token in parsed:
        if sentence is None or token.is_sent_start:
            if not sentence is None:
                yield sentence
            sentence = []
        sentence.append(token)

    if not sentence is None and len(sentence) > 0:
        yield sentence

def getAugmentedDependencyPath(sentence, G, assoc_tokens, a, b):
	nicest_path,indexest_path = None,None
	for i,j in itertools.product(assoc_tokens[a],assoc_tokens[b]):
		path = shortest_path(G, i, j)
        #print(path)

		index_path, nice_path = [], []
		for index in range(len(path)):
			#nice_path.append(sentence[path[index]].text)
			node_label = G.nodes[path[index]]['label']

			# Remove the actual entity from the begin and end node labels in the path
			if index == 0 or index == (len(path)-1):
				node_label = '~'.join(['*'] + node_label.split('~')[1:])

			nice_path.append(node_label)
			if index < len(path)-1:
				nice_path.append(G.edges[(path[index],path[index+1])]['label'])
				index_path.append(path[index])

		#nice_path = nice_path[1:]
		#index_path = index_path[1:]

		if not nicest_path or len(nice_path) < len(nicest_path):
			nicest_path = nice_path
			indexest_path = index_path

	return nicest_path

def formatSentence(sentence,passage,assoc_tokens,a,b):
	sentence_start = sentence[0].idx
	sentence_end = sentence[-1].idx + len(sentence[-1].text)
	sentence_text = passage.text[sentence_start:sentence_end]

	a_start = sentence[assoc_tokens[a][0]].idx - sentence_start
	a_end = sentence[assoc_tokens[a][-1]].idx + len(sentence[assoc_tokens[a][-1]].text) - sentence_start
	a_type = a.data.infons['type']

	b_start = sentence[assoc_tokens[b][0]].idx - sentence_start
	b_end = sentence[assoc_tokens[b][-1]].idx + len(sentence[assoc_tokens[b][-1]].text) - sentence_start
	b_type = b.data.infons['type']

	char_array = [ html.escape(c) for c in sentence_text ]
	
	char_array[a_start] = '<entity1>' + char_array[a_start]
	char_array[a_end-1] = char_array[a_end-1] + '</entity1>'
	char_array[b_start] = '<entity2>' + char_array[b_start]
	char_array[b_end-1] = char_array[b_end-1] + '</entity2>'

	formatted_sentence = "".join(char_array)

	return formatted_sentence
	
def checkRelation(entity1, entity2, relation, passage):
	start1 = entity1.locations[0].offset - passage.offset
	start2 = entity2.locations[0].offset - passage.offset

	end1 = start1 + entity1.locations[0].length
	end2 = start2 + entity2.locations[0].length

	deppath = relation.infons['deppath']

	text_after_entity1 = passage.text[end1:]
	text_after_entity2 = passage.text[end2:]

	# Remove relations with entities that precede +
	if text_after_entity1.startswith('+') or text_after_entity2.startswith('+'):
		return False

	# Remove relations with entities that are neighbouring words
	text_between_entities = passage.text[end1:start2] if end1 <= start2 else passage.text[end2:start1]
	if text_between_entities.strip() == '':
		return False

	# Remove some obvious more complex situations where there's an extra event (e.g. induction)
	# that's not appearing in the dependency path
	after_words_to_skip = ['stimulated','induced','mediated']
	for w in after_words_to_skip:
		if not w in deppath and (text_after_entity1.startswith(w) or text_after_entity2.startswith(w)):
			return False

	return True

	

def processSentence(sentence,passage,tree):

	depparse_tuples = []
	token_to_index = { token:i for i,token in enumerate(sentence) }
	for i,token in enumerate(sentence):
		if token.head in token_to_index:
			j = token_to_index[token.head]
			label = token.dep_
			#print(sentence[i], sentence[j], label)
			depparse_tuples.append((i,j,label))

	all_nmods = { i:j for i,j,label in depparse_tuples if label == 'nmod' }

	all_amods = defaultdict(list)
	all_cases = defaultdict(list)
	for i,j,label in depparse_tuples:
		if label == 'amod':
			all_amods[j].append(i)
		elif label == 'case':
			all_cases[j].append(i)
	#all_amods = { j:i for i,j,label in depparse_tuples if label == 'amod' }

	all_negs = set( j for i,j,label in depparse_tuples if label == 'neg' )

	token_texts = [ t.text for t in sentence ]

	#sentence_start = sentence[0].idx
	#sentence_end = sentence[-1].idx + len(sentence[-1].text)
	#sentence_text = passage.text[sentence_start:sentence_end]

	check_all_parents_for_negs = False

	if check_all_parents_for_negs:

		for i,token in enumerate(sentence):
			#print(token.text, token.pos_, token.dep_, token.head)
			if token.pos_ == 'VERB':
				polarity = True
				cur_token = token
				while True:
					j = token_to_index[cur_token]
					if j in all_negs:
						polarity = not polarity
					#print(' ',cur_token, j, all_negs, polarity)

					if cur_token.dep_ == 'ROOT':
						break
					cur_token = cur_token.head

				if polarity:
					token_texts[i] = "+%s" % token.text
				else:
					token_texts[i] = "-%s" % token.text

	else:
		for i,token in enumerate(sentence):
			if token.pos_ == 'VERB':
				polarity = not (i in all_negs)

				if polarity:
					token_texts[i] = "+%s" % token.text
				else:
					token_texts[i] = "-%s" % token.text

	prev_words_to_attach = {'si','depletion','knockdown','silenced','anti','anti-'}

	G = nx.Graph()
	for i,token_text in enumerate(token_texts):
		prev_word = None
		if i > 0:
			prev_word = token_texts[i-1].lower()

		#print(i,token_text)
		token_with_modifiers = token_text.lower()
		if i in all_nmods:
			token_with_modifiers += '~%s' % token_texts[all_nmods[i]].lower()
		if i in all_amods:
			token_with_modifiers += '~' + '~'.join( [ token_texts[j].lower() for j in all_amods[i] ] )
		if i in all_cases:
			token_with_modifiers += '~' + '~'.join( [ token_texts[j].lower() for j in all_cases[i] ] )
		if prev_word in prev_words_to_attach:
			token_with_modifiers += '~%s' % prev_word

		#print(i, token_with_modifiers)
		G.add_node(i,label=token_with_modifiers)

	#print("all_nmods:", all_nmods)
	#print("all_amods:", all_amods)

	# We tweak the dependency parse so that elements with nmod/amod parents move their
	# children to their parent
	for i,j,label in depparse_tuples:
		#if j in all_nmods and label != 'compound':
		#	j = all_nmods[j]
		#if i in all_amods and label != 'compound':
		#	i = all_amods[i]

		#print(i,j,label)

		G.add_edge(i,j,label=label.lower())

	token_boundaries = set( [ t.idx for t in sentence ] + [ (t.idx+len(t.text)) for t in sentence ] )

	assoc_tokens = defaultdict(list)
	for i,token in enumerate(sentence):
		start,end = (token.idx, token.idx+len(token.text))
		for a in tree[start:end]:
			# Only allow annotations that match up with token boundaries
			# so no sub-words, unless the tokenizer can see it
			if a.begin in token_boundaries and a.end in token_boundaries:
				assoc_tokens[a].append(i)

	relations = []

	sentence_start_with_offset = passage.offset + sentence[0].idx
	sentence_end_with_offset = passage.offset + sentence[-1].idx + len(sentence[-1].text)

	for a,b in itertools.product(assoc_tokens,assoc_tokens):
		if a != b and a.data.infons['type'] <= b.data.infons['type']:

			if a.data.infons['type'] != 'Gene' or b.data.infons['type'] != 'Gene':
				continue

			dep_path_steps = getAugmentedDependencyPath(sentence, G, assoc_tokens, a, b)

			deppath_is_symmetrical = (dep_path_steps == dep_path_steps[::-1])
			contains_conj = 'conj' in dep_path_steps
			
			if len(dep_path_steps) >= 3 and len(dep_path_steps) <= 9 and not deppath_is_symmetrical and not contains_conj:
				aug_dep_path = '|'.join(dep_path_steps)

				formatted_sentence = formatSentence(sentence,passage,assoc_tokens,a,b)
				assert not '\t' in formatted_sentence

				relation = bioc.BioCRelation()
				relation.infons['deppath'] = aug_dep_path
				relation.infons['deppath_length'] = len(dep_path_steps)
				relation.infons['sentence_start'] = sentence_start_with_offset
				relation.infons['sentence_end'] = sentence_end_with_offset
				relation.infons['formatted_sentence'] = formatted_sentence

				relation.add_node(bioc.BioCNode(refid=a.data.id,role='entity1'))
				relation.add_node(bioc.BioCNode(refid=b.data.id,role='entity2'))
				#relation.add_node(b.data)

				if checkRelation(a.data,b.data,relation,passage):
					passage.add_relation(relation)


				#out_data = [ str(doc.infons['pmid']), a.data.infons['type'], a.data.infons['conceptid'], b.data.infons['type'], b.data.infons['conceptid'], str(len(dep_path_steps)), aug_dep_path, formatted_sentence ]
				#outF.write('\t'.join(out_data) + '\n')
				#assert False

	return relations
	
def processDoc(doc):
	hyphen_plus_word = re.compile(r'-[a-z]+\b')

	for passage in doc.passages:
		if len(passage.annotations) == 0:
			continue

		passage.text = passage.text.replace('\r', ' ')
		passage.text = passage.text.replace('\n', ' ')
		passage.text = passage.text.replace('\t', ' ')
	
		# Strip out buggy microRNA tags (where the text is just mir)
		# and also LPS (as it's always lipopolysaccharide and not a gene
		passage.annotations = [ a for a in passage.annotations if a.text.lower() not in ['mir','lps'] ]

		tree = IntervalTree()
		for a in passage.annotations:
			start = a.locations[0].offset - passage.offset
			end = start + a.locations[0].length

			# Fill in empty parts (for debug data)
			if not 'conceptid' in a.infons:
				a.infons['conceptid'] = '-'

			following_text = passage.text[end:]
			if hyphen_plus_word.match(following_text):
				# Replace the hyphen with a space
				passage.text = passage.text[:end] + ' ' + passage.text[(end+1):]

			tree.addi(start,end,a)

		for sentence in sentencesGenerator(passage.text):
			sentence_start = sentence[0].idx
			sentence_end = sentence[-1].idx + len(sentence[-1].text)
			sentence_length = sentence_end - sentence_start
			sentence_text = passage.text[sentence_start:sentence_end]

			semicomma_count = sentence_text.count(';')

			# Do some basic filtering to remove odd cases
			if sentence_length < 1000 and semicomma_count < 5:
				#print(sentence_text)
				#processSentence(sentence,passage,doc,tree,outF)
				processSentence(sentence,passage,tree)


	# Renumber relations
	relationNo = 0
	for p in doc.passages:
		for r in p.relations:
			relationNo += 1
			r.id = "R%d" % relationNo


def main():
	parser = argparse.ArgumentParser(description='Find all dependency paths between entities in same sentences')
	parser.add_argument('--inBioC',required=True,type=str,help='PubTator annotated documents')
	parser.add_argument('--mode',required=True,type=str,help='Whether to output a table-delimited file (tsv) or the BioC file (bioc)')
	parser.add_argument('--outFile',required=True,type=str,help='Output file')
	args = parser.parse_args()

	args.mode = args.mode.lower()
	assert args.mode.lower() in ['tsv','bioc']
	
	with open(args.inBioC,'rb') as inF:
		parser = bioc.BioCXMLDocumentReader(inF)

		if args.mode == 'bioc':
			writer = bioc.BioCXMLDocumentWriter(args.outFile)
		elif args.mode == 'tsv':
			writer = open(args.outFile,'w')

		for i,doc in enumerate(parser):
			#if doc.infons['pmid'] != '31388315':
			#	continue

			# Skip documents with PMIDs
			if not ('pmid' in doc.infons and doc.infons['pmid']):
				continue
				
			processDoc(doc)

			if args.mode == 'bioc':
				writer.write_document(doc)
			else:
				for p in doc.passages:
					annotation_map = { a.id:a for a in p.annotations }

					for r in p.relations:
						e1 = annotation_map[r.nodes[0].refid]
						e2 = annotation_map[r.nodes[1].refid]

						out_data = [ doc.infons['pmid'], e1.infons['type'], e1.infons['conceptid'], e2.infons['type'], e2.infons['conceptid'], r.infons['deppath_length'], r.infons['deppath'], r.infons['formatted_sentence'] ]
						writer.write('\t'.join(map(str,out_data)) + '\n')
				writer.flush()

		writer.close()
			
	print("Done")

if __name__ == '__main__':
	main()

