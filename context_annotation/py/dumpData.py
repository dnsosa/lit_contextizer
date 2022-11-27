import mysql.connector
from collections import defaultdict,Counter
import json
import argparse
import re
import gzip
import bioc
		
def gunzip_bytes_obj(bytes_obj: bytes) -> str:
	return gzip.decompress(bytes_obj).decode()
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser('Dump the annotations out to a file')
	parser.add_argument('--db',required=True,type=str,help='JSON with database settings')
	parser.add_argument('--inBioc',required=True,type=str,help='BioC file to add annotations to')
	parser.add_argument('--outBioc',required=True,type=str,help='BioC output file with the documents and new annotations')
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
	#print(pmid_to_documentid)
		
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
		
	sql = "SELECT document_id,pmid,contents FROM documents"
	mycursor.execute(sql)
	myresult = mycursor.fetchall()
	
	writer = bioc.BioCXMLDocumentWriter(args.outBioc)
	
	written_count = 0
	breakdown = Counter()
	
	parser = bioc.BioCXMLDocumentReader(args.inBioc)
	for doc in parser:
		pmid = int(doc.infons['pmid'])
		#print(pmid)
		assert pmid in pmid_to_documentid
		document_id = pmid_to_documentid[pmid]
		
		bib_to_pmid = {}
		for p in doc.passages:
			for a in p.annotations:
				if a.infons['type'] == 'xref' and 'pmid' in a.infons and a.text:
					bib_to_pmid[a.text.strip(' []()')] = a.infons['pmid']
				#print(a.infons)
		#print(bib_to_pmid)
		#assert False
		
		unannotated = False
		for p in doc.passages:
			new_relations = []
			
			for r in p.relations:
				notes = relation_notes[(document_id,r.id)] if (document_id,r.id) in relation_notes else ''
				annos = previous_annotations[pmid][r.id]
				
				cite = 'citation:'
				if 'error' in notes:
					pass
				elif 'citation:' in notes:
					r.infons['is_citing'] = 'True'
					citations = notes[(notes.index(cite)+len(cite)):]
					citations = citations.split('+')
					citation_pmids = [ bib_to_pmid[b.strip()] for b in citations if b.strip() in bib_to_pmid ]
					#print(citation_pmids)
					
					r.infons['citation_pmids'] = ",".join(citation_pmids)
					
					new_relations.append(r)
				elif len(annos) > 0:
					r.infons['is_citing'] = 'False'
					
					contexts = "|".join( f"{concept_type}:{source_context_id}" for concept_type,source_context_id in annos )
					r.infons['contexts'] = contexts
					
					new_relations.append(r)
				else:
					unannotated = True
					
			p.relations = new_relations
			
		relation_count = len( [ r for p in doc.passages for r in p.relations ] )
					
		if relation_count > 0 and not unannotated:
			has_citing = any( r.infons['is_citing'] == 'True' for p in doc.passages for r in p.relations )
			has_contexts = any( r.infons['is_citing'] == 'False' for p in doc.passages for r in p.relations )
			
			if has_citing and has_contexts:
				breakdown['citing and contexts'] += 1
			elif has_citing:
				breakdown['citing'] += 1
			elif has_contexts:
				breakdown['contexts'] += 1
			else:
				breakdown['neither'] += 1
		
			written_count += 1
			writer.write_document(doc)
	
	print("Written %d documents" % written_count)
	print(breakdown)
	