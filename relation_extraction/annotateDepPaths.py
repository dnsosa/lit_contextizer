import argparse
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from collections import Counter,OrderedDict
import os

def main():
	parser = argparse.ArgumentParser(description='Find nearest ten for an embedding')
	parser.add_argument('--embeddings',required=True,type=str,help='File with embeddings')
	parser.add_argument('--deppath_counts',required=True,type=str,help='Deppath counts file')
	parser.add_argument('--annotations',required=True,type=str,help='Annotations file (for loading and saving)')
	parser.add_argument('--seed',required=True,type=str,help='Base deppath to search for')
	parser.add_argument('--top',required=False,type=int,default=1000,help='Chose the closest N deppaths to seed and order by frequency')
	parser.add_argument('--search',required=False,type=str,help='Search string to use on dependency paths')
	parser.add_argument('--metric',required=False,default='euclidean',type=str,help='Distance metric to use')
	args = parser.parse_args()

	annotations = OrderedDict()
	if os.path.isfile(args.annotations):
		print("Loading existing annotations")
		with open(args.annotations) as f:
			for line in f:
				tag,deppath,example_sentence = line.strip('\n').split('\t')
				annotations[deppath] = (tag,example_sentence)

	print("Loading embeddings...")

	with open(args.embeddings) as f:
		num_columns = len(f.readline().split('\t'))

	embedding_cols = list(range(2,num_columns))
	embeddings = np.loadtxt(args.embeddings,comments=None,delimiter='\t',usecols=embedding_cols)

	selected_index = None
	with open(args.embeddings) as f:
		deppath_data = []
		for i,line in enumerate(f):
			split = line.strip('\n').split('\t')
			#print(split)
			deppath,example_sentence = split[:2]
			deppath_data.append( (deppath,example_sentence) )

			if deppath == args.seed:
				selected_index = i

	print("Filtering out deppaths with 'conj'...")
	nonconj_deppath_indices = [ i for i,(deppath,example_sentence) in enumerate(deppath_data) if not 'conj' in deppath.split('|') ]
	deppath_data = [ deppath_data[i] for i in nonconj_deppath_indices ]
	embeddings = embeddings[nonconj_deppath_indices,:]

	print("Finding seed deppath...")
	search = [ i for i,(deppath,example_sentence) in enumerate(deppath_data) if deppath == args.seed ]
	assert len(search) == 1, "Could not find deppath %s in data" % args.seed
	selected_embedding = embeddings[search[0],:].reshape(1,-1)

	print("Calculating distances...")
	assert args.metric in ['euclidean','cosine']
	if args.metric == 'cosine':
		normalize(selected_embedding)
		normalize(embeddings)
	distances = pairwise_distances(selected_embedding, embeddings, metric=args.metric)
	distances = distances.flatten().tolist()
	distances_with_index = sorted(list(enumerate(distances)), key=lambda x: x[1])

	top_choices = distances_with_index[:args.top]
	top_choice_deppaths = set( deppath_data[index][0] for index,d in top_choices )

	print("Loading deppath frequency counts...")
	deppath_counts = Counter()
	with open(args.deppath_counts) as f:
		for line in f:
			count,deppath,example_sentence = line.strip('\n').split('\t')
			if deppath in top_choice_deppaths:
				deppath_counts[deppath] = int(count)

	top_choices_with_counts = [ (deppath_counts[deppath_data[index][0]],d,index) for index,d in top_choices ]
	top_choices_with_counts = sorted(top_choices_with_counts, key=lambda x: (-x[0],x[1]))

	print("####################")
	print("# Annotation Time! #")
	print("####################")

	for rank,(count,d,index) in enumerate(top_choices_with_counts):
		deppath, example_sentence = deppath_data[index]
		if deppath in annotations:
			continue

		if args.search and not args.search in deppath:
			continue

		deppath_split = deppath.split('|')
		deppath_is_symmetrical = (deppath_split == deppath_split[::-1])
		if deppath_is_symmetrical:
			continue # Skip it as we need directionality

		out = [ str(rank), str(count), "%.2e" % d, deppath, example_sentence ]
		print()
		print("\t".join(out))
		tag = input("Annotation? ")

		annotations[deppath] = (tag,example_sentence)

		with open(args.annotations,'w') as outF:
			for deppath,(tag,example_sentence) in annotations.items():
				outF.write("%s\t%s\t%s\n" % (tag,deppath,example_sentence))
	


if __name__ == '__main__':
	main()

