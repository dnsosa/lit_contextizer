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
	
	
def gzip_str(string_: str) -> bytes:
	out = io.BytesIO()

	with gzip.GzipFile(fileobj=out, mode='w') as fo:
		fo.write(string_.encode())

	bytes_obj = out.getvalue()
	return bytes_obj

		
if __name__ == '__main__':
	parser = argparse.ArgumentParser('Integrate in same documents with new annotations, keeping existing annotations')
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
	
	sql = "SELECT document_id,pmid FROM documents"
	mycursor.execute(sql)
	myresult = mycursor.fetchall()
	
	pmid_to_documentid = {}
	for document_id,pmid in myresult:
		pmid_to_documentid[pmid] = document_id
		
	sql = "SELECT relation_id,source_relation_id,document_id,notes FROM relations"
	mycursor.execute(sql)
	myresult = mycursor.fetchall()
	
	expected_relations = defaultdict(list)
	relation_notes = {}
	relation_id_map = {}
	for relation_id,source_relation_id,document_id,notes in myresult:
		expected_relations[document_id].append(source_relation_id)
		relation_id_map[(document_id,source_relation_id)] = relation_id
		if notes:
			relation_notes[(document_id,source_relation_id)] = notes
			
			
	sql = "SELECT d.pmid,r.source_relation_id,r.document_id,c.source_context_id,c.name,c.type FROM documents d, relations r, annotations a, contexts c WHERE d.document_id = r.document_id AND r.relation_id = a.relation_id AND a.context_id = c.context_id"
	mycursor.execute(sql)
	myresult = mycursor.fetchall()
	
	previous_annotations = defaultdict(lambda: defaultdict(list))
	previous_annotation_names = {}
	for pmid,source_relation_id,document_id,source_context_id,concept_name,concept_type in myresult:
	
		previous_annotation_names[(pmid,concept_type,source_context_id)] = concept_name
		previous_annotations[pmid][source_relation_id].append((concept_type,source_context_id))
		#print(row)
			
	#print(relation_notes)
	#print(pmid_to_documentid)
	#print(annotations)

	
	
	#context_types = set( ['Disease','CellLine','Chemical','Species'] )
	context_types = set( ['CellContext','Species'] )
	
	all_contexts = {}
	
	doc_update_sql = 'UPDATE documents SET contents = %s WHERE document_id = %s'
	doc_update_records = []
	
	parser = bioc.BioCXMLDocumentReader(args.bioc)
	for doc in parser:
		pmid = int(doc.infons['pmid'])
		assert pmid in pmid_to_documentid
		document_id = pmid_to_documentid[pmid]
	
		context_mentions = [ a for p in doc.passages for a in p.annotations if a.infons['type'] in context_types ]
		
		context_names = defaultdict(Counter)
		for a in context_mentions:
			concept_id = a.infons['conceptid']
			concept_type = a.infons['type']
			#print(pmid,concept_id,concept_type)
			context_names[(concept_type,concept_id)][a.text.lower()] += 1
			
		context_names = { (concept_type,concept_id):counts.most_common(1)[0][0] for (concept_type,concept_id),counts in context_names.items() }
		#contexts = set(context_names.keys())
	
		all_contexts[pmid] = context_names
		
		
		collection = bioc.BioCCollection()
		collection.add_document(doc)
		contents = bioc.dumps(collection)
		compressed = gzip_str(contents)
		
		doc_update_records.append((compressed,document_id))
	
		relation_ids = [ r.id for r in doc.relations ]
		assert sorted(relation_ids) == sorted(expected_relations[pmid]), "Relations don't match for document PMID=%s (%s != %s)"% (pmid,sorted(relation_ids),sorted(expected_relations[pmid]))
		#for a in doc.annotations:
		
	
	for pmid in previous_annotations:
		for source_relation_id,associated_contexts in previous_annotations[pmid].items():
			for concept_type,concept_id in associated_contexts:
				concept_name = previous_annotation_names[(pmid,concept_type,concept_id)]
				#print(concept_type,concept_id)
				#print(all_contexts[pmid])
				assert (concept_type,concept_id) in all_contexts[pmid], f"Could't find matching context for PMID={pmid} with name={concept_name}, type={concept_type} and id={concept_id}"

	context_sql = 'INSERT INTO contexts(context_id,source_context_id,document_id,name,type) VALUES(%s,%s,%s,%s,%s)'
	context_records = []
	context_ids = {}
	
	context_id = 1
	for pmid in all_contexts:
		document_id = pmid_to_documentid[pmid]
		for (concept_type,concept_id),concept_name in all_contexts[pmid].items():
			context_records.append( (context_id,concept_id,document_id,concept_name,concept_type) )
			
			context_ids[ (document_id,concept_type,concept_id) ] = context_id
			
			context_id += 1
			
	#previous_annotations[pmid][source_relation_id].append((concept_type,source_context_id))
	annotations_sql = 'INSERT INTO annotations(relation_id,context_id,added) VALUES(%s,%s,NOW())'
	annotation_records = []
	for pmid in previous_annotations:
		document_id = pmid_to_documentid[pmid]
		for source_relation_id,associated_contexts in previous_annotations[pmid].items():
			relation_id = relation_id_map[(document_id,source_relation_id)]
		
			for concept_type,concept_id in associated_contexts:
				context_id = context_ids[ (document_id,concept_type,concept_id) ]
				annotation_records.append( (relation_id,context_id) )
			
	
	mycursor.execute("DELETE FROM contexts")
	mycursor.execute("DELETE FROM annotations")
	
	#mycursor.executemany(doc_update_sql, doc_update_records)
	
	mycursor.executemany(context_sql, context_records)
	mycursor.executemany(annotations_sql, annotation_records)
	
	#print(context_records)
	#print(annotation_records)
	
	#assert False
	#mydb.commit()
