import argparse
import sqlite3
import sys
import os
import shutil

def mergeDBs(input_dbs,output_db,truncate_inputs=False):
	assert isinstance(input_dbs,list), "Expected list of input DB files"
	assert isinstance(output_db, str), "Expected string with output DB"

	# If the output_db doesn't exist, use the first input db and merge into it
	if not os.path.isfile(output_db):
		print("Starting with %s..." % input_dbs[0])
		shutil.copyfile(input_dbs[0],output_db)
		input_dbs = input_dbs[1:]

	con = sqlite3.connect(output_db)
	cur = con.cursor()

	for input_db in input_dbs:
		assert os.path.getsize(input_db) > 0, "Input db file (%s) is empty" % input_db

		print("Processing %s..." % input_db)
		sys.stdout.flush()

		cur.execute("ATTACH DATABASE ? as input_db ;", (input_db, ))

		cur.execute(f"REPLACE INTO metadata SELECT * FROM input_db.metadata;")
		con.commit()

		cur.execute("DETACH DATABASE input_db ;")
		
		con.commit()

	con.close()

def main():
	parser = argparse.ArgumentParser('Merge a set of databases')
	parser.add_argument('--inDir',required=True,type=str,help='Input directory with DBs')
	parser.add_argument('--outDB',required=True,type=str,help='Output SQLite DB')
	args = parser.parse_args()

	input_dbs = sorted( os.path.join(args.inDir,f) for f in os.listdir(args.inDir) if f.endswith('.db') )

	mergeDBs(input_dbs, args.outDB)

	print("Done.")

if __name__ == '__main__':
	main()

