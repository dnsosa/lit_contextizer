import argparse
import itertools
from collections import Counter,defaultdict
import sys
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
import numpy as np
from sklearn.manifold import TSNE

def main():
	parser = argparse.ArgumentParser(description='Do an SVD on a id-pairs/dep-path matrix')
	parser.add_argument('--inFile',required=True,type=str,help='Input matrix file')
	parser.add_argument('--k',required=False,type=int,default=8,help='Rank of low-rank SVD')
	parser.add_argument('--outFile',required=True,type=str,help='Output stuff')
	args = parser.parse_args()

	indices_i, indices_j, values = [],[],[]

	print("Loading matrix...")
	sys.stdout.flush()
	
	index_to_deppath_and_sentence = {}
	
	with open(args.inFile) as f:
		for line in f:
			split = line.strip('\n').split('\t')
			i,j,id1_id2,deppath,count,example_sentence = split
			
			i,j = int(i),int(j)
			
			index_to_deppath_and_sentence[j] = (deppath,example_sentence)
			
			indices_i.append(i)
			indices_j.append(j)
			values.append(1.0)
			
	num_rows = max(indices_i)+1
	num_cols = max(indices_j)+1
			
	sparse = coo_matrix((values,(indices_i, indices_j)), shape=(num_rows,num_cols))
	sparse = sparse.tocsr()
	indices_i, indices_j, values = [],[],[]
	
	print("Loaded sparse matrix of size:",sparse.shape)
	
	print("Running SVD...")
	sys.stdout.flush()
	
	U, sing, V = svds(sparse, args.k)
	
	print("U.shape=",U.shape)
	print("sing.shape=",sing.shape)
	print("V.shape=",V.shape)
	
	print("Multiplying sqrt(sing) with V...")
	# Spreading singular values
	V = np.diag(np.sqrt(sing)).dot(V)
	#print(V.shape)

	#print("Calculating T-SNE...")
	#embeddings = TSNE(n_components=2).fit_transform(V.T)
	#print("embeddings.shape=",embeddings.shape)
	#assert num_cols == embeddings.shape[0]
	
	print("Saving V...")
	
	with open(args.outFile,'w') as outF:
		for i in range(num_cols):
			deppath,example_sentence = index_to_deppath_and_sentence[i]
			#vector = embeddings[i,:].flatten().tolist()
			vector = V[:,i].flatten().tolist()
			
			#print(V[:,i])
			
			#break
			
			out_data = [deppath,example_sentence] + list(map(str,vector))
			outF.write("\t".join(out_data) + "\n")
			
	print("Done")
	
	
if __name__ == '__main__':
	main()
	
