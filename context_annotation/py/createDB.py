import mysql.connector
from collections import OrderedDict
import csv
import json
import argparse

def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser('Create tables in database (and will remove old data!)')
	parser.add_argument('--db',required=True,type=str,help='JSON with database settings')
	args = parser.parse_args()
	
	with open(args.db) as f:
		database = json.load(f)

	mydb = mysql.connector.connect(
	  host=database['host'],
	  user=database['user'],
	  passwd=database['passwd'],
	  database=database['database']
	)
	
	engine = "MyISAM"
		
	mycursor = mydb.cursor()

	mycursor.execute("DROP TABLE IF EXISTS relations")
	mycursor.execute("DROP TABLE IF EXISTS documents")
	mycursor.execute("DROP TABLE IF EXISTS annotations")
	mycursor.execute("DROP TABLE IF EXISTS contexts")

	columns = OrderedDict()
	columns['relation_id'] = 'INT NOT NULL AUTO_INCREMENT'
	columns['source_relation_id'] = 'VARCHAR(255)'
	columns['document_id'] = 'INT'
	columns['sentence'] = 'TEXT'
	columns['entity1'] = 'VARCHAR(255)'
	columns['entity2'] = 'VARCHAR(255)'
	columns['index_in_doc'] = 'INT'
	columns['total_in_doc'] = 'INT'
	columns['notes'] = 'TEXT'

	fields = ", ".join("%s %s" % (n,t) for n,t in columns.items())
	fields += ", PRIMARY KEY(%s)" % list(columns.keys())[0]
	sql = "CREATE TABLE relations (%s) ENGINE=%s" % (fields,engine)
	print(sql)
	mycursor.execute(sql)
	
	columns = OrderedDict()
	columns['document_id'] = 'INT NOT NULL AUTO_INCREMENT'
	columns['pmid'] = 'INT'
	columns['pmcid'] = 'INT'
	columns['doi'] = 'VARCHAR(255)'
	columns['title'] = 'TEXT'
	columns['journal'] = 'VARCHAR(255)'
	columns['year'] = 'INT'
	columns['contents'] = 'BLOB'

	fields = ", ".join("%s %s" % (n,t) for n,t in columns.items())
	fields += ", PRIMARY KEY(%s)" % list(columns.keys())[0]
	sql = "CREATE TABLE documents (%s) ENGINE=%s" % (fields,engine)
	print(sql)
	mycursor.execute(sql)

	columns = OrderedDict()
	columns['annotation_id'] = 'INT NOT NULL AUTO_INCREMENT'
	columns['relation_id'] = 'INT'
	columns['context_id'] = 'INT'
	columns['added'] = 'DATETIME'

	fields = ", ".join("%s %s" % (n,t) for n,t in columns.items())
	fields += ", PRIMARY KEY(%s)" % list(columns.keys())[0]
	sql = "CREATE TABLE annotations (%s) ENGINE=%s" % (fields,engine)
	print(sql)
	mycursor.execute(sql)
	

	columns = OrderedDict()
	columns['context_id'] = 'INT NOT NULL AUTO_INCREMENT'
	columns['source_context_id'] = 'VARCHAR(255)'
	columns['document_id'] = 'INT'
	columns['name'] = 'VARCHAR(255)'
	columns['type'] = 'VARCHAR(255)'

	fields = ", ".join("%s %s" % (n,t) for n,t in columns.items())
	fields += ", PRIMARY KEY(%s)" % list(columns.keys())[0]
	sql = "CREATE TABLE contexts (%s) ENGINE=%s" % (fields,engine)
	print(sql)
	mycursor.execute(sql)
	
	mydb.commit()
