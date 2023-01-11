import argparse
import sqlite3
import sys
import os
import xml.etree.cElementTree as etree
import io
import gzip

def gzip_str(string_: str) -> bytes:
	out = io.BytesIO()

	with gzip.GzipFile(fileobj=out, mode='w') as fo:
		fo.write(string_.encode())

	bytes_obj = out.getvalue()
	return bytes_obj

def main():
	parser = argparse.ArgumentParser('Create a database of metadata info')
	parser.add_argument('--inBioc',required=True,type=str,help='Input BioC file')
	parser.add_argument('--outDB',required=True,type=str,help='Output SQLite DB')
	args = parser.parse_args()


	con = sqlite3.connect(args.outDB)
	
	cur = con.cursor()
	cur.execute("CREATE TABLE metadata(pmid INTEGER PRIMARY KEY ASC, compressed BLOB);")
	con.commit()

	#input_files = [ f for f in os.listdir(args.inDir) if f.startswith('pubmed') and f.endswith('.bioc.xml') ]
	#input_files = sorted(input_files, reverse=True)
	seen_pmids = set()

	#for input_file in input_files:
	#	print(input_file)
	#	sys.stdout.flush()

	metadata_records = []
	#with open(os.path.join(args.inDir,input_file)) as f:
	with open(args.inBioc) as f:
		for event, elem in etree.iterparse(f, events=('start', 'end', 'start-ns', 'end-ns')):
			if (event=='end' and elem.tag=='document'):

				pmid_field = elem.find('./id')

				pmid = None
				if pmid_field is not None and pmid_field.text and pmid_field.text != 'None':
					pmid = int(pmid_field.text)

				if pmid and not pmid in seen_pmids:
					seen_pmids.add(pmid)

					metadata_fields = elem.findall('./infon')
					metadata_xmls = [ etree.tostring(mf, encoding='utf8', method='html').decode() for mf in metadata_fields ]
					metadata_singlexml = "<infons>%s</infons>" % "".join(metadata_xmls)
					compressed = gzip_str(metadata_singlexml)

					metadata_record = (pmid, compressed)
					metadata_records.append(metadata_record)

				elem.clear()

	cur.executemany("INSERT INTO metadata VALUES (?,?)", metadata_records)

	con.commit()

	con.close()

	print("Done.")

if __name__ == '__main__':
	main()

