import argparse
import numpy as np
import json
from termcolor import colored
from collections import OrderedDict
import os
import random

class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKCYAN = '\033[96m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'


def decorateSentence(sentence):
	chars = list(sentence['text'])

	start1,end1 = sentence['entity1_location']
	start2,end2 = sentence['entity2_location']

	chars[start1] = bcolors.FAIL + chars[start1]
	chars[end1-1] += bcolors.ENDC

	chars[start2] = bcolors.OKBLUE + chars[start2]
	chars[end2-1] += bcolors.ENDC
	
	return "".join(chars)

def main():
	parser = argparse.ArgumentParser(description='Try to learn a classifier for up/down regulation from BAI sentences')
	parser.add_argument('--sentenceJSON',required=True,type=str,help='Sentence JSON data')
	#parser.add_argument('--sentenceVectors',required=True,type=str,help='Sentence vectors as numpy matrix')
	parser.add_argument('--annotations',required=True,type=str,help='Input/output annotation data')
	args = parser.parse_args()

	with open(args.sentenceJSON) as f:
		sentences = json.load(f)

	#matrix = np.load(args.sentenceVectors,allow_pickle=True)

	annotations = OrderedDict()
	if os.path.isfile(args.annotations):
		with open(args.annotations) as f:
			for line in f:
				index, anno, decorated = line.strip('\n').split('\t')
				index = int(index)
				annotations[index] = anno

	while True:
		index = random.randint(0, len(sentences)-1)
		if index in annotations:
			continue

		sentence = sentences[index]
		decorated = decorateSentence(sentence)

		print()
		print("[%d]" % (len(annotations)+1), decorated)
		anno = input('? ')
		annotations[index] = anno

		with open(args.annotations,'w') as outF:
			for index,anno in annotations.items():
				decorated = decorateSentence(sentences[index])
				outF.write("%d\t%s\t%s\n" % (index,anno,decorated))


if __name__ == '__main__':
	main()
