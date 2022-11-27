import SPARQLWrapper
import argparse
from collections import defaultdict
import json
import re

def runQuery(query):
	endpoint = 'https://query.wikidata.org/sparql'
	sparql = SPARQLWrapper.SPARQLWrapper(endpoint, agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36')
	sparql.setQuery(query)
	sparql.setReturnFormat(SPARQLWrapper.JSON)
	results = sparql.query().convert()

	return results['results']['bindings']
	
def main():
	parser = argparse.ArgumentParser(description='Tool to pull cell types, lines, and tissue types from WikiData using SPARQL')
	parser.add_argument('--stopwords',type=str,required=True,help='File with stopwords')
	parser.add_argument('--outTSV',type=str,required=True,help='File to output entities')
	args = parser.parse_args()

	with open(args.stopwords) as f:
		stopwords = set( line.strip().lower() for line in f )

	totalCount = 0
	
	groups = {'cell type':'Q189118','cell line':'Q21014462','tissue':'Q40397'}

	entities = defaultdict(dict)

	for groupType,groupID in groups.items():
	
		print("Gathering data from Wikidata for %s..." % groupType)
			
		query = """
		SELECT ?entity ?entityLabel ?entityDescription ?alias WHERE {
			SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
			?entity wdt:P31 wd:%s.
			OPTIONAL {?entity skos:altLabel ?alias FILTER (LANG (?alias) = "en") .}
		} 

		""" % groupID

		rowCount = 0
		for row in runQuery(query):
			longID = row['entity']['value']
			
			if 'xml:lang' in row['entityLabel'] and row['entityLabel']['xml:lang'] == 'en':
			
				# Get the Wikidata ID, not the whole URL
				shortID = longID.split('/')[-1]
						
				entity = entities[shortID]
				entity['id'] = shortID
				entity['name'] = row['entityLabel']['value']
				entity['type'] = groupType
							
				if 'entityDescription' in row and 'xml:lang' in row['entityDescription'] and row['entityDescription']['xml:lang'] == 'en':
					entity['description'] = row['entityDescription']['value']
				
				if not 'aliases' in entity:
					entity['aliases'] = []

				if 'alias' in row and row['alias']['xml:lang'] == 'en':
					entity['aliases'].append(row['alias']['value'])

			rowCount += 1
			totalCount += 1

	for entityID,entity in entities.items():
		entity['aliases'].append(entity['name'])
		entity['aliases'] = [ t.strip().lower() for t in entity['aliases'] ]
		entity['aliases'] = [ t for t in entity['aliases'] if len(t) > 2 ]
		entity['aliases'] = [ t for t in entity['aliases'] if not t in stopwords ]
		entity['aliases'] += [ t.replace('\N{REGISTERED SIGN}','').strip() for t in entity['aliases'] ]
		entity['aliases'] += [ t + 's' for t in entity['aliases'] if t.endswith(' cell') ]
		entity['aliases'] = [ t for t in entity['aliases'] if not '/' in t ]
		entity['aliases'] = [ t for t in entity['aliases'] if not re.match(r'^\d+$',t) ]
		entity['aliases'] = sorted(set(entity['aliases']))
			
	entities = { entityID:entity for entityID,entity in entities.items() if len(entity['aliases']) > 0 }
	
	# Remove entities with '/' in their name
	entities = { entityID:entity for entityID,entity in entities.items() if not '/' in entity['name'] }	
	
	print ("  Got %d entities (from %d rows)" % (len(entities),totalCount))

	print("Saving TSV file...")
	with open(args.outTSV,'w') as f:
		for entityID in sorted(entities.keys()):
			entity = entities[entityID]
			outData = [ entity['id'], entity['name'], '|'.join(entity['aliases']), entity['type'] ]
			f.write("\t".join(outData) + "\n")

if __name__ == '__main__':
	main()

