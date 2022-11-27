import argparse
import itertools
from collections import Counter,defaultdict
import sys

def iterateFile(filename):
	with open(filename) as f:
		for line in f:
			split = line.strip('\n').split('\t')

			pmid, id1, id2, deppath_length, deppath, sentence = split

			if id1 == '' or id1 == '-' or id2 == '' or id2 == '-':
				continue

			deppath_length = int(deppath_length)
			if deppath_length < 3:
				continue

			yield pmid, id1, id2, deppath_length, deppath, sentence

def main():
	parser = argparse.ArgumentParser(description='Count the frequency of dependency paths')
	parser.add_argument('--inFile',required=True,type=str,help='Input file with entities, dependency paths and sentences')
	parser.add_argument('--deppath_threshold',required=False,type=int,default=10,help='Threshold to prune deppaths')
	parser.add_argument('--outFile',required=True,type=str,help='Output file')
	args = parser.parse_args()
	
	#pair_counts = Counter()
	#deppath_counts = Counter()
	
	print("Loading file searching for frequent deppaths...")
	sys.stdout.flush()
	deppath_counts = Counter()
	example_sentences = {}
	for pmid, id1, id2, deppath_length, deppath, sentence in iterateFile(args.inFile):
		deppath_counts[deppath] += 1

	deppath_counts = { deppath:count for deppath,count in deppath_counts.items() if count >= args.deppath_threshold }

	print("Getting example sentences...")
	sys.stdout.flush()
	example_sentences = {}
	for pmid, id1, id2, deppath_length, deppath, sentence in iterateFile(args.inFile):
		if deppath in deppath_counts and (not deppath in example_sentences or len(sentence) < len(example_sentences[deppath])):
			example_sentences[deppath] = sentence

	deppaths_above_threshold = [ (count,deppath) for deppath,count in deppath_counts.items() if count >= args.deppath_threshold ]
	deppaths_above_threshold = sorted(deppaths_above_threshold, reverse=True)

	print("Saving...")
	sys.stdout.flush()

	with open(args.outFile,'w') as outF:
		for count,deppath in deppaths_above_threshold:
			example_sentence = example_sentences[deppath]
			outF.write("%d\t%s\t%s\n" % (count,deppath,example_sentence))
	
	print("Done.")
	sys.stdout.flush()
	
	
if __name__ == '__main__':
	main()
	
