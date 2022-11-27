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
	parser = argparse.ArgumentParser(description='Make the sparse deppath/entity-pair matrix')
	parser.add_argument('--inFile',required=True,type=str,help='Input file with entities, dependency paths and sentences')
	parser.add_argument('--cooccur_threshold',required=False,type=int,default=10,help='Threshold to prune id_pairs')
	parser.add_argument('--deppath_threshold',required=False,type=int,default=10,help='Threshold to prune deppaths')
	parser.add_argument('--degree_threshold',required=False,type=int,default=10,help='Threshold to prune nodes from the matrix')
	parser.add_argument('--outFile',required=True,type=str,help='Output file')
	args = parser.parse_args()
	

	
	#pair_counts = Counter()
	#deppath_counts = Counter()
	
	print("Loading file searching for frequent id_pairs")
	sys.stdout.flush()
	pair_counts = Counter()
	for pmid, id1, id2, deppath_length, deppath, sentence in iterateFile(args.inFile):
		id_pair = (id1,id2)
		pair_counts[id_pair] += 1
			
	pairs_above_threshold = set( id_pair for id_pair,cooccur_count in pair_counts.items() if cooccur_count >= args.cooccur_threshold )
	pair_counts = None

	print("Loading file searching for frequent deppaths")
	sys.stdout.flush()
	deppath_counts = Counter()
	for pmid, id1, id2, deppath_length, deppath, sentence in iterateFile(args.inFile):
		deppath_counts[deppath] += 1
			
	deppaths_above_threshold = set( deppath for deppath,count in deppath_counts.items() if count >= args.deppath_threshold )
	deppath_counts = None
	
	print("Loading file with matrix using only frequent id_pairs and deppaths")
	sys.stdout.flush()


	matrix = defaultdict(Counter)
	for pmid, id1, id2, deppath_length, deppath, sentence in iterateFile(args.inFile):
		id_pair = (id1,id2)
		if id_pair in pairs_above_threshold and deppath in deppaths_above_threshold:
			matrix[id_pair][deppath] += 1
			
			# Get a short example sentence for each deppath
			#if not deppath in example_sentences or len(sentence) < len(example_sentences[deppath]):
			#	example_sentences[deppath] = sentence
			
			#pair_counts[id_pair] += 1
			#deppath_counts[deppath] += 1

	#print("Removing uncommon id_pairs with threshold=%d...", args.cooccur_threshold)
	#pair_counts = [ (id_pair,sum(matrix[id_pair].values())) for id_pair in matrix ]

	#matrix = { id_pair:deppath_data for id_pair,deppath_data in matrix.items() if id_pair in pairs_above_threshold }
	#print("Reduced pair count from %d to %d" % (len(pair_counts),len(pairs_above_threshold)))
	
	print("Pruning matrix nodes with degree threshold=%d..." % args.degree_threshold)
	sys.stdout.flush()
			
	while True:
		#all_id_pairs = set(matrix.keys())
		#all_deppaths = set( deppath for id_pair in matrix for deppath in matrix[id_pair] )
		
		print("  Transposing matrix...")
		sys.stdout.flush()
		
		matrix_transpose = defaultdict(Counter)
		for id_pair in matrix:
			for deppath in matrix[id_pair]:
				matrix_transpose[deppath][id_pair] = matrix[id_pair][deppath]
		
		print("  Gathering counts...")
		sys.stdout.flush()
	
		#pair_counts = { id_pair:sum(matrix[id_pair].values()) for id_pair in matrix }
		#deppath_counts = sum( [ matrix[id_pair] for id_pair in matrix ], Counter())
		
		pair_counts = { id_pair:len(matrix[id_pair]) for id_pair in matrix }
		deppath_counts = { deppath:len(matrix_transpose[deppath]) for deppath in matrix_transpose }
		
		
		#deppath_counts = sum( [ matrix[id_pair] for id_pair in matrix ], defaultdict(set))
		
		print("  Trimming id-pairs/deppaths...")
		sys.stdout.flush()
	
		id_pairs = set( id_pair for id_pair,count in pair_counts.items() if count >= args.degree_threshold )
		deppaths = set( deppath for deppath,count in deppath_counts.items() if count >= args.degree_threshold )
		
		print("  Calculating matrix size...")
		sys.stdout.flush()
		
		before_count = sum( 1 for id_pair in matrix for deppath in matrix[id_pair] )
		
		print("  Reforming matrix...")
		sys.stdout.flush()
		
		#matrix = { id_pair:matrix[id_pair] for id_pair in matrix if id_pair in id_pairs }
		matrix = { id_pair:Counter({deppath:count for deppath,count in matrix[id_pair].items() if deppath in deppaths }) for id_pair in matrix if id_pair in id_pairs }
		
		print("  Recalculating matrix size...")
		sys.stdout.flush()
		
		after_count = sum( 1 for id_pair in matrix for deppath in matrix[id_pair] )
		
		print("Pruning: before=%d after=%d" % (before_count, after_count))
		sys.stdout.flush()
		if before_count == after_count:
			# No change
			break
		
	print("Getting example sentences...")
	sys.stdout.flush()
	example_sentences = {}
	for pmid, id1, id2, deppath_length, deppath, sentence in iterateFile(args.inFile):
		id_pair = (id1,id2)
		if deppath in deppaths:
			if not deppath in example_sentences or len(sentence) < len(example_sentences[deppath]):
				example_sentences[deppath] = sentence

	print("Calculating indices...")
	sys.stdout.flush()
	
	id_pairs = sorted(id_pairs)
	deppaths = sorted(deppaths)
	
	id_pair_2_index = { id_pair:i for i,id_pair in enumerate(id_pairs) }
	deppath_2_index = { deppath:j for j,deppath in enumerate(deppaths) }

	print("Saving...")
	sys.stdout.flush()
	
	with open(args.outFile,'w') as outF:
		#for (i,id_pair),(j,deppath) in itertools.product(enumerate(id_pairs),enumerate(deppaths)):
		for id_pair in matrix:
			for deppath in matrix[id_pair]:
				id1,id2 = id_pair
				
				i,j = id_pair_2_index[id_pair], deppath_2_index[deppath]
				
				count = matrix[id_pair][deppath]
				example_sentence = example_sentences[deppath]
				
				outF.write("%d\t%d\t%s|%s\t%s\t%d\t%s\n" % (i,j,id1,id2,deppath,count,example_sentence))

	print("Output %dx%d matrix" % (len(id_pairs),len(deppaths)))
	
if __name__ == '__main__':
	main()
	
