import sys, os, random
'''
 script to encode stdin captions wrt a given vocabulary 
 '''
if len(sys.argv) < 3:
	print("Usage: %s <vocab> word1 word2 ..." % sys.argv[0])
	exit(1)


vocab_file = sys.argv[1]
annot_file = sys.argv[2]

words = sys.argv[2:] 


# read vocab
vocab = []
with open(vocab_file,'r') as f:
	for line in f:
		vocab.append(line.strip())

unks = [ w for w in words if not w in vocab]
words = [w if w in vocab else 'UNK' for w in words]

if unks:
	print("Generated %d UNK mappings:" % len(unks))
	for u in unks:
		print("%s -> UNK" % u)


# print encoding
print("Caption: %s" % str(words))
print("Indexes: %s" % " ".join(map(str,[ vocab.index(w) for w in words])))
