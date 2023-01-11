import argparse
import sqlite3
import sys
import os
import shutil
import xml.etree.cElementTree as ET
import gzip

def gunzip_bytes_obj(bytes_obj: bytes) -> str:
	return gzip.decompress(bytes_obj).decode()

def mergeInMetadata(document_root, metadata_root):
	assert list(document_root)[0].tag == 'id', "Expected first tag of document to be the id"

	# Removing the metadata from the full text document
	for infon in document_root.findall('./infon'):
		document_root.remove(infon)

	# Copying over the metadata from the PubMed/abstract to the full text document
	for infon in reversed(metadata_root.findall('./infon')):
		document_root.insert(1,infon)

	#xmlstr = ET.tostring(fulltext_root, encoding='utf8', method='html').decode()
	#return xmlstr

def main():
	parser = argparse.ArgumentParser('Integrate metadata into documents from a metadata DB')
	parser.add_argument('--inBioc',required=True,type=str,help='Input BioC file')
	parser.add_argument('--outBioc',required=True,type=str,help='Output BioC file')
	parser.add_argument('--db',required=True,type=str,help='Metadata DB')
	args = parser.parse_args()

	with open(args.inBioc) as f:
		tree = ET.parse(f)

	#tree = ET.parse(args.inBioc)
	root = tree.getroot()

	con = sqlite3.connect(args.db)
	cur = con.cursor()
	
	for document in root.findall('./document'):
		pmid_field = document.find('./id')

		pmid = None
		if pmid_field is not None and pmid_field.text and pmid_field.text != 'None':
			pmid = int(pmid_field.text)

			cur.execute('SELECT compressed FROM metadata WHERE pmid = ?', (pmid,))
			metadata = cur.fetchone()
			if metadata:
				metadata = gunzip_bytes_obj(metadata[0])

				meta_root = ET.fromstring(metadata)

				mergeInMetadata(document,meta_root)
				
			#print(pmid)
		#break

	con.close()

	with open(args.outBioc, 'wb') as f:
		tree.write(f)

	print("Done.")

if __name__ == '__main__':
	main()

