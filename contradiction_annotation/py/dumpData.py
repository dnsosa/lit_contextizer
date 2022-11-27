import mysql.connector
from collections import defaultdict,Counter
import json
import argparse
import re
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser('Dump the annotations out to a file')
	parser.add_argument('--corpus_name',required=True,type=str,help='Name of the corpus')
	parser.add_argument('--db',required=True,type=str,help='JSON with database settings')
	parser.add_argument('--outJSON',required=True,type=str,help='JSON file')
	args = parser.parse_args()
	
	with open(args.db) as f:
		database = json.load(f)

	mydb = mysql.connector.connect(
	  host=database['host'],
	  user=database['user'],
	  passwd=database['passwd'],
	  database=database['database']
	)
	mycursor = mydb.cursor(dictionary=True)

	sql = "SELECT corpus_id FROM corpora WHERE name=%s"
	mycursor.execute(sql, (args.corpus_name,))
	myresult = mycursor.fetchall()
	assert len(myresult) == 1
	corpus_id = myresult[0]['corpus_id']
	
	pairs = {}
	
	sql = "SELECT pair_id,as_json,notes FROM pairs WHERE corpus_id=%s"
	mycursor.execute(sql, (corpus_id,))
	myresult = mycursor.fetchall()
	for row in myresult:
		pair_id = row['pair_id']
		notes = row['notes']
		
		decoded = json.loads(row['as_json'])
		decoded = { k:v if v != 'None' else None for k,v in decoded.items() }		
		decoded['notes'] = row['notes']
		#decoded['pair_id'] = pair_id
		decoded['annotations'] = []
		
		pairs[pair_id] = decoded
		
		#print(decoded)
		
	sql = "SELECT a.pair_id, o.name FROM annotations a, options o WHERE a.option_id = o.option_id"
	mycursor.execute(sql)
	myresult = mycursor.fetchall()
	for row in myresult:
		pair_id = row['pair_id']
		name = row['name']
		
		if pair_id in pairs:
			pairs[pair_id]['annotations'].append(name)
	
		#print(row)
		
	output = [ pair_data for pair_id,pair_data in pairs.items() if pair_data['annotations'] ]
	
	for d in output:
		d['is_valid'] = any( a for a in d['annotations'] if a.startswith('Different') )
		
		notes = d['notes']
		if not notes:
			notes = ''
		
		reasoning = {}
		
		for field in ['up','down','up_citation','down_citation']:
			match = re.match(r"\b%s:(?P<val>\S*)\b" % field,notes)
			reasoning[field] = None
			if match:
				reasoning[field] = match.groupdict()['val'].replace('_',' ')
				notes = notes[:match.start()] + notes[match.end():]
				
			notes = notes.strip()
		
		d['notes'] = notes
		d['reasoning'] = reasoning
		print(notes, reasoning)
		
	print(Counter( a for d in output for a in d['annotations'] ))
	
	valid_count = len([ d for d in output if d['is_valid'] ])
		
	with open(args.outJSON,'w',encoding='utf8') as f:
		json.dump(output,f,indent=2,sort_keys=True)
	print("Saved %d pairs to file with %d valid" % (len(output),valid_count))
	