import argparse
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

def main():
	parser = argparse.ArgumentParser(description='Find nearest ten for an embedding')
	parser.add_argument('--embeddings',required=True,type=str,help='File with embeddings')
	parser.add_argument('--deppath',required=True,type=str,help='Deppath to search for')
	parser.add_argument('--metric',required=False,default='euclidean',type=str,help='Distance metric to use')
	parser.add_argument('--top',required=False,default=10,type=int,help='Number to show')
	args = parser.parse_args()

	with open(args.embeddings) as f:
		num_columns = len(f.readline().split('\t'))

	embedding_cols = list(range(2,num_columns))
	embeddings = np.loadtxt(args.embeddings,comments=None,delimiter='\t',usecols=embedding_cols)
	print(embeddings.shape)

	selected_index = None
	with open(args.embeddings) as f:
		deppath_data = []
		for i,line in enumerate(f):
			split = line.strip('\n').split('\t')
			#print(split)
			deppath,example_sentence = split[:2]
			deppath_data.append( (deppath,example_sentence) )

			if deppath == args.deppath:
				selected_index = i

	assert not selected_index is None, "Could not find deppath %s in data" % args.deppath

	selected_embedding = embeddings[selected_index,:].reshape(1,-1)

	assert args.metric in ['euclidean','cosine']

	if args.metric == 'cosine':
		#selected_embedding = normalize(selected_embedding)
		#embeddings = normalize(embeddings)
		normalize(selected_embedding)
		normalize(embeddings)

	distances = pairwise_distances(selected_embedding, embeddings, metric=args.metric)
	print(distances.shape)

	distances = distances.flatten().tolist()

	sort_descending = False #args.metric == 'cosine'

	distances_with_index = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=sort_descending)

	for index,d in distances_with_index[:args.top]:
		deppath, example_sentence = deppath_data[index]
		out = [ "%.2e" % d, deppath, example_sentence ]
		print("\t".join(out))

	


if __name__ == '__main__':
	main()

