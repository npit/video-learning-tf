import sys, os, random
'''
 script to visualize captions of a processed, dataset given the dataset and vocabulary files
 '''
if len(sys.argv) < 3:
	print("Usage: %s <vocab> <annotation> <opt1 opt2>" % sys.argv[0])
	print("opts: random max <maxnum>")
	exit(1)


vocab_file = sys.argv[1]
annot_file = sys.argv[2]

opts = sys.argv[3:] if len(sys.argv) > 3 else []
randomize = False
maxnum = None
if 'random' in opts:
	randomize = True
if 'max' in opts:
	i = opts.index('max')
	if len(opts) >=  i+2:
		try:
			maxnum = int(opts[i+1])
		except:
			maxnum = 10
			print('%s is not a number for max, using %d' % (opts[i+1], maxnum))

# read vocab
vocab = []
with open(vocab_file,'r') as f:
	for line in f:
		vocab.append(line.strip())

# read image annotations
images_captions = []
with open(annot_file,'r') as f:
	for line in f:
		elements = line.strip().split()

		images_captions.append((elements[0], list(map(int,elements[1:]))))



if randomize:
	print("Randomizing input order.")
	random.shuffle(images_captions)
if maxnum is not None:
	print("Restricting input instances from %d to %d." % (len(images_captions),maxnum))
	images_captions = images_captions[:maxnum]

for i,pair in enumerate(images_captions):
	caption = [ vocab[v] for v in pair[1]]
	print("Image %d/%d : [%s] - [%s]" % (i+1,len(images_captions),pair[0], " ".join(caption)))