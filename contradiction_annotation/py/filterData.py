import mysql.connector
from collections import defaultdict,Counter
import csv
import json
import argparse
from datetime import datetime, date
import calendar
import csv
import random
import re

def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser('Load pair data into database')
	parser.add_argument('--data',required=True,type=str,help='CSV file with BAI data')
	#parser.add_argument('--docs',required=True,type=str,help='JSON documents for ID mapping')
	parser.add_argument('--outdata',required=True,type=str,help='CSV file with filtered data')
	args = parser.parse_args()
	
	
	
	pairs = set()
	data = defaultdict(list)
	
	id_to_likely_name = defaultdict(Counter)
	
	sign_flip = {'+':'-', '-':'+'}
	
	sign_to_num = {'+':1, '-':-1}
	num_to_sign = {1:'+', -1:'-'}
	
	skip_words = ['signaling','signalling','pathway']
	
	common_side_words = Counter()
	skip_side_words = ['proliferation','complex','ubiquitination','phosphorylation','cleavage','dephosphorylation','ubiquitylation','regulation','recruitment','family','signaling','signalling','pathway','translocation','binding','administration','acetylation','localization','neutralization','processing','polyubiquitination','axis','subunit','inflammasome','member','endocytosis','dimerization','sumoylation','methylation','dimer','component','precursor','residue','superfamily','domain','heterodimer','homodimer','ligand','region','response','stabilization','secretion','release']
	
	skip_predicates = ['affect','affects']
	
	with open(args.data, 'r', encoding='utf-8' ) as f:
		reader = csv.DictReader(f)
		for row in reader:
			pair = (row['entity1_uuid'],row['entity2_uuid'])
			pairs.add(pair)
			
			row['entity1_text'] = row['entity1_text'].strip()
			row['entity2_text'] = row['entity2_text'].strip()
			
			if row['entity1_uuid'] == row['entity2_uuid']:
				continue
			
			id_to_likely_name[row['entity1_uuid']][row['entity1_text']] += 1
			id_to_likely_name[row['entity2_uuid']][row['entity2_text']] += 1
			
			side_words = []
			for haystack in [row['entity1_text'],row['entity2_text']]:
				word = re.search(' [a-z]+$',haystack)
				if word and len(word.group()) != len(haystack):
					word = word.group().strip()
					side_words.append(word)
					if not word in skip_side_words:
						common_side_words[word] += 1
					
			# Removing CD as they are more often cell types than genes/proteins (a bit simplistic)
			#if re.search('\bCD\d+', row['entity1_text']):
			#	continue
			#if re.search('\bCD\d+', row['entity2_text']):
			#	continue
			#for subword in row['entity1_text'].split() + row['entity2_text'].split():
				
			looks_like_celltype = False
			looks_like_mediated = False
			for e in [1,2]:
				entity_text = row['entity%d_text' % e].lower()
				start_position = int(row['entity%d_begin_index' % e])
				end_position = int(row['entity%d_end_index' % e])
				
				before_text = row['text'][:start_position].strip().lower()
				after_text = row['text'][end_position:].strip().lower()
				
				if re.match(r"\s*[\+]\s+",after_text):
					looks_like_celltype = True
				if after_text.startswith('-mediated'):
					looks_like_mediated = True
				if after_text.startswith('-induced'):
					looks_like_mediated = True
					
				if before_text.endswith('inhibition of') and 'inhibition' not in entity_text:
					row['entity%d_sign' % e] = sign_flip[row['entity%d_sign' % e]]
				if before_text.endswith('anti-') and 'anti' not in entity_text:
					row['entity%d_sign' % e] = sign_flip[row['entity%d_sign' % e]]
				if before_text.endswith('silenced') and 'silence' not in entity_text:
					row['entity%d_sign' % e] = sign_flip[row['entity%d_sign' % e]]
					
			if looks_like_celltype:
				pass
			if looks_like_mediated:
				continue
			
			if any(w in row['text'] for w in skip_words):
				continue
			if any(w in side_words for w in skip_side_words):
				continue
				
			if row['predicate'] in skip_predicates:
				continue
			
			#print(" | ".join([row['overall_sign'],row['entity1_text'],row['entity2_text'],row['text']]))
			
			row['overall_sign'] = num_to_sign[sign_to_num[row['entity1_sign']] * sign_to_num[row['verb_sign']] * sign_to_num[row['entity2_sign']]]
			
			row['doc_title'] = row['doc_title_string']
			row['journal'] = row['journal_string']
			
			
			data[(row['entity1_uuid'],row['entity2_uuid'],row['overall_sign'])].append(row)
			
	id_to_likely_name = { id:counter.most_common(1)[0][0] for id,counter in id_to_likely_name.items() }
	
	#for word,count in common_side_words.most_common(100):
	#	print("%d\t%s" % (count,word))
	#print(common_side_words.most_common(100))
			
	#print(f"len(data) = {len(data)}")
	
	all_rows = sum(data.values(),[])
	print(f"len(all_rows) = {len(all_rows)}")
	
	for row in random.sample(all_rows,10):
		print(" | ".join([row['overall_sign'],row['entity1_text'],row['entity2_text'],row['text']]))
		print()
		
			
	random.seed(42)
	pair_count = 0
	
	with open(args.outdata,'w',encoding='utf8',newline='') as outF:
		writer = csv.DictWriter(outF, all_rows[0].keys())
		writer.writeheader()
		for entity1_uuid,entity2_uuid in sorted(pairs):
			if len(data[(entity1_uuid,entity2_uuid,'+')]) >= 2 and len(data[(entity1_uuid,entity2_uuid,'-')]) >= 2:
				for row in data[(entity1_uuid,entity2_uuid,'+')] + data[(entity1_uuid,entity2_uuid,'-')]:
					writer.writerow(row)
				
				#data_u = random.choice(data[(entity1_uuid,entity2_uuid,'+')])
				#data_d = random.choice(data[(entity1_uuid,entity2_uuid,'-')])
				
				#print(" | ".join([data_u['entity1_text'],data_u['entity2_text'],data_u['text']]))
				#print()
				#assert False
			
				pair_count += 1
		
	print(f"pair_count = {pair_count}")