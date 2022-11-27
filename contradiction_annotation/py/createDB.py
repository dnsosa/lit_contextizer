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

	mycursor.execute("DROP TABLE IF EXISTS pairs")
	mycursor.execute("DROP TABLE IF EXISTS annotations")
	mycursor.execute("DROP TABLE IF EXISTS options")

	columns = OrderedDict()
	columns['pair_id'] = 'INT NOT NULL AUTO_INCREMENT'
	columns['corpus_id'] = 'INT'
	columns['entity1'] = 'VARCHAR(255)'
	columns['entity2'] = 'VARCHAR(255)'
	
	columns['a_entity1_text'] = 'VARCHAR(255)'
	columns['a_entity2_text'] = 'VARCHAR(255)'
	columns['a_entity1_begin'] = 'INT'
	columns['a_entity1_end'] = 'INT'
	columns['a_entity2_begin'] = 'INT'
	columns['a_entity2_end'] = 'INT'
	columns['a_simplified_text'] = 'TEXT'
	columns['a_text'] = 'TEXT'
	columns['a_year'] = 'INT'
	columns['a_month'] = 'INT'
	columns['a_section_name'] = 'VARCHAR(255)'
	columns['a_doc_title'] = 'VARCHAR(255)'
	columns['a_journal'] = 'VARCHAR(255) NULL'
	columns['a_pmid'] = 'INT'
	columns['a_pmcid'] = 'INT'
	columns['a_doi'] = 'VARCHAR(255)'
	
	columns['b_entity1_text'] = 'VARCHAR(255)'
	columns['b_entity2_text'] = 'VARCHAR(255)'
	columns['b_entity1_begin'] = 'INT'
	columns['b_entity1_end'] = 'INT'
	columns['b_entity2_begin'] = 'INT'
	columns['b_entity2_end'] = 'INT'
	columns['b_simplified_text'] = 'TEXT'
	columns['b_text'] = 'TEXT'
	columns['b_year'] = 'INT'
	columns['b_month'] = 'INT'
	columns['b_section_name'] = 'VARCHAR(255)'
	columns['b_doc_title'] = 'VARCHAR(255)'
	columns['b_journal'] = 'VARCHAR(255) NULL'
	columns['b_pmid'] = 'INT'
	columns['b_pmcid'] = 'INT'
	columns['b_doi'] = 'VARCHAR(255)'
	
	columns['as_json'] = 'TEXT'
	columns['notes'] = 'TEXT'

	fields = ", ".join("%s %s" % (n,t) for n,t in columns.items())
	fields += ", PRIMARY KEY(%s)" % list(columns.keys())[0]
	sql = "CREATE TABLE pairs (%s) ENGINE=%s" % (fields,engine)
	print(sql)
	mycursor.execute(sql)

	columns = OrderedDict()
	columns['annotation_id'] = 'INT NOT NULL AUTO_INCREMENT'
	columns['pair_id'] = 'INT'
	columns['option_id'] = 'INT'
	columns['added'] = 'DATETIME'

	fields = ", ".join("%s %s" % (n,t) for n,t in columns.items())
	fields += ", PRIMARY KEY(%s)" % list(columns.keys())[0]
	sql = "CREATE TABLE annotations (%s) ENGINE=%s" % (fields,engine)
	print(sql)
	mycursor.execute(sql)
	

	columns = OrderedDict()
	columns['option_id'] = 'INT NOT NULL AUTO_INCREMENT'
	columns['name'] = 'VARCHAR(255)'

	fields = ", ".join("%s %s" % (n,t) for n,t in columns.items())
	fields += ", PRIMARY KEY(%s)" % list(columns.keys())[0]
	fields += ", INDEX(name)"
	sql = "CREATE TABLE options (%s) ENGINE=%s" % (fields,engine)
	print(sql)
	mycursor.execute(sql)
	
	
	columns = OrderedDict()
	columns['corpus_id'] = 'INT NOT NULL AUTO_INCREMENT'
	columns['name'] = 'VARCHAR(255)'

	fields = ", ".join("%s %s" % (n,t) for n,t in columns.items())
	fields += ", PRIMARY KEY(%s)" % list(columns.keys())[0]
	fields += ", INDEX(name)"
	sql = "CREATE TABLE corpora (%s) ENGINE=%s" % (fields,engine)
	print(sql)
	mycursor.execute(sql)
	
	insertsql = "INSERT INTO options(option_id,name) VALUES (%s,%s)"
	option_records = [ (1,'Skip'), (2,'Mistake'), (3,'Same Context'), (4,'Different Disease'), (5,'Different Cell Type') ]
	mycursor.executemany(insertsql, option_records)

	mydb.commit()
