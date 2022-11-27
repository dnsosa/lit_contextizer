import mysql.connector
from collections import defaultdict,OrderedDict,Counter
import csv
import json
import argparse
import bioc

#from io import StringIO
import gzip
import io

def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]
		
def gzip_str(string_: str) -> bytes:
	out = io.BytesIO()

	with gzip.GzipFile(fileobj=out, mode='w') as fo:
		fo.write(string_.encode())

	bytes_obj = out.getvalue()
	return bytes_obj


def gunzip_bytes_obj(bytes_obj: bytes) -> str:
	return gzip.decompress(bytes_obj).decode()

		
if __name__ == '__main__':
	parser = argparse.ArgumentParser('Load relations for context annotation')
	parser.add_argument('--db',required=True,type=str,help='JSON with database settings')
	parser.add_argument('--bioc',required=True,type=str,help='BioC file with relations to be annotated')
	args = parser.parse_args()
	
	with open(args.db) as f:
		database = json.load(f)

	mydb = mysql.connector.connect(
	  host=database['host'],
	  user=database['user'],
	  passwd=database['passwd'],
	  database=database['database']
	)
		
	mycursor = mydb.cursor()

	parser = bioc.BioCXMLDocumentReader(args.bioc)
	
	#context_types = set( ['Disease','CellLine','Chemical','Species'] )
	context_types = set( ['CellContext','Species'] )
	
	for docno,doc in enumerate(parser):
		print(docno)
		document_id = docno + 1
		pmid = int(doc.infons['pmid'])
		pmcid = doc.infons['pmcid'] if 'pmcid' in doc.infons and doc.infons['pmcid'] else None
		doi = doc.infons['doi'] if 'doi' in doc.infons and doc.infons['doi'] else None
	
		relations = [ r for p in doc.passages for r in p.relations ]
		
		annotations = [ a for p in doc.passages for a in p.annotations ]
		
		annotation_map = {}
		for a in annotations:
			annotation_map[a.id] = a
		
		contexts = [ a for a in annotations if a.infons['type'] in context_types ]
		
		context_names = defaultdict(Counter)
		for a in contexts:
			conceptid = a.infons['conceptid']
			concepttype = a.infons['type']
			
			context_names[(concepttype,conceptid)][a.text.lower()] += 1
			
		context_names = { (concepttype,conceptid):counts.most_common(1)[0][0] for (concepttype,conceptid),counts in context_names.items() }
		
		collection = bioc.BioCCollection()
		collection.add_document(doc)
		contents = bioc.dumps(collection)
		
		compressed = gzip_str(contents)
		
		document_sql = 'INSERT INTO documents(document_id,pmid,pmcid,doi,title,journal,year,contents) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)'
		document_record = (document_id, pmid, pmcid, doi, doc.infons['title'], doc.infons['journal'], doc.infons['year'], compressed)
		mycursor.execute(document_sql, document_record)
		
		relation_sql = 'INSERT INTO relations(source_relation_id,document_id,sentence,entity1,entity2,index_in_doc,total_in_doc) VALUES(%s,%s,%s,%s,%s,%s,%s)'
		relation_records = []
		for relno,r in enumerate(relations):
			entity1 = annotation_map[r.nodes[0].refid].text
			entity2 = annotation_map[r.nodes[1].refid].text
		
			relation_record = (r.id,document_id,r.infons['formatted_sentence'],entity1,entity2,relno+1,len(relations))
			relation_records.append(relation_record)
		mycursor.executemany(relation_sql, relation_records)
		
		context_sql = 'INSERT INTO contexts(source_context_id,document_id,name,type) VALUES(%s,%s,%s,%s)'
		context_records = []
		for (concepttype,conceptid),conceptname in context_names.items():
			context_record = (conceptid,document_id,conceptname,concepttype)
			context_records.append(context_record)
		mycursor.executemany(context_sql, context_records)
			
		#print(context_names)

	mydb.commit()
