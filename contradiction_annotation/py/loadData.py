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
	parser.add_argument('--corpus_name',required=True,type=str,help='Name for the corpus')
	parser.add_argument('--db',required=True,type=str,help='JSON with database settings')
	args = parser.parse_args()
	
	
	if False:
		doi_to_other_ids = {}
		with open(args.docs,encoding='utf8') as f:
			for line in f:
				d = json.loads(line)
				id_split = d['doc_id'].split('|')
				
				assert len(id_split) == 3 or len(id_split) == 1
				if len(id_split) == 3:
					pmid,pmcid,doi = id_split
					if pmid:
						pmid = int(pmid.replace('PM',''))
					else:
						pmid = None
					if pmcid:
						pmcid = int(pmcid.replace('PMC',''))
					else:
						pmcid = None
				else:
					doi = id_split
					pmid,pmcid = None,None
				
				doi_to_other_ids[d['doi']] = (pmid,pmcid)
			
		print("Loaded IDs from %d documents" % len(doi_to_other_ids))
	
	with open(args.db) as f:
		database = json.load(f)

	mydb = mysql.connector.connect(
	  host=database['host'],
	  user=database['user'],
	  passwd=database['passwd'],
	  database=database['database']
	)

	mycursor = mydb.cursor()
	
	pairs = set()
	data = defaultdict(list)
	
	id_to_likely_name = defaultdict(Counter)
	
	insertsql = "INSERT INTO corpora(name) VALUES(%s)"
	mycursor.execute(insertsql, (args.corpus_name,))
	corpus_id = mycursor.lastrowid
	print("Added corpus with name: %s (id=%d)" % (args.corpus_name,corpus_id))
	
	with open(args.data, 'r', encoding='utf-8' ) as f:
		reader = csv.DictReader(f,delimiter='\t')
		for row in reader:
			#print(row)
			row['entity1_text'] = row['entity1_text'].strip()
			row['entity2_text'] = row['entity2_text'].strip()
			
			pair = (row['entity1_uuid'],row['entity2_uuid'])
			pairs.add(pair)
			
			#if not row['entity1_uuid'] in id_to_likely_name or len(row['entity1_text']) < len(id_to_likely_name[row['entity1_uuid']]):
			#	id_to_likely_name[row['entity1_uuid']] = row['entity1_text']
			#if not row['entity2_uuid'] in id_to_likely_name or len(row['entity2_text']) < len(id_to_likely_name[row['entity2_uuid']]):
			#	id_to_likely_name[row['entity2_uuid']] = row['entity2_text']
				
			id_to_likely_name[row['entity1_uuid']][row['entity1_text']] += 1
			id_to_likely_name[row['entity2_uuid']][row['entity2_text']] += 1
			
			#row['doc_title'] = row['doc_title_string']
			#row['journal'] = row['journal_string']
			
			row['entity1_begin'] = row['entity1_begin_index']
			row['entity1_end'] = row['entity1_end_index']
			row['entity2_begin'] = row['entity2_begin_index']
			row['entity2_end'] = row['entity2_end_index']
			
			#del row['doc_title_string']
			#del row['journal_string']
			del row['entity1_begin_index']
			del row['entity1_end_index']
			del row['entity2_begin_index']
			del row['entity2_end_index']
			
			#assert row['doi'] in doi_to_other_ids
			#row['pmid'],row['pmcid'] = doi_to_other_ids[row['doi']]
			
			data[(row['entity1_uuid'],row['entity2_uuid'],row['overall_sign'])].append(row)
			
	id_to_likely_name = { id:counter.most_common(1)[0][0] for id,counter in id_to_likely_name.items() }
			
	fields = ['entity1_text','entity2_text','entity1_begin','entity1_end','entity2_begin','entity2_end','text','year','month','section_name','doc_title','journal','pmid','pmcid','doi']
	
	
	columns = ['corpus_id','entity1','entity2', 'as_json'] + [f"a_{f}" for f in fields ]  + [f"b_{f}" for f in fields ]
	dbfields = ",".join(columns)
	dbvalues = ",".join('%s' for _ in columns)
	insertsql = "INSERT INTO pairs (%s) VALUES (%s)" % (dbfields,dbvalues)
			
	random.seed(42)
	insertrecords = []
	for entity1_uuid,entity2_uuid in sorted(pairs):
		data_u = random.choice(data[(entity1_uuid,entity2_uuid,'+')])
		data_d = random.choice(data[(entity1_uuid,entity2_uuid,'-')])
		
		data_to_insert = {'entity1':id_to_likely_name[entity1_uuid],'entity2':id_to_likely_name[entity2_uuid]}
		#for f,v in data_u.items():
		for f in fields:
			data_to_insert[f'a_{f}'] = data_u[f]
		for f in fields:
			data_to_insert[f'b_{f}'] = data_d[f]
			
		data_to_insert['as_json'] = json.dumps({'up':data_u, 'down':data_d})
			
		data_to_insert['corpus_id'] = corpus_id
		
		insertrecord = [ data_to_insert[c] for c in columns ]
		insertrecords.append(insertrecord)
		
	random.seed(42)
	random.shuffle(insertrecords)
		
	for chunk in chunks(insertrecords, 100):
		mycursor.executemany(insertsql, chunk)
		
	mydb.commit()
	print("Added %d pairs" % len(insertrecords))
		
	
		