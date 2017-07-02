import numpy as np
import sys
from utils_ import *
from process_annotations import read_vocabulary

'''
Script to encode vocabulary contents into the embedding matrix. 
'''
init_file = "config.ini"
# vocabulary and embeddings files
vocabulary_file = None
embeddings_file_type = "glove"
embeddings_file = "/path/to/embeddings"

if len(sys.argv) > 1:
    init_file = sys.argv[1]
print("Using initialization file : ",init_file)

keyvals = init_config(init_file, "captions")

for key in keyvals:
    exec("%s=%s" % (key, keyvals[key]))
print("Successfully initialized from file %s" % init_file)

vocab = read_vocabulary(vocabulary_file)

if embeddings_file_type == "glove":
    embeddings = {}
    embedding_minmax = [10000, -10000]
    # read the whole contents
    print("Reading embeddings file...",end="")
    with open(embeddings_file,"r") as fp:
        for line in fp:
            contents = line.split()
            token = contents[0]
            vector = list(map(float, contents[1:]))
            embeddings[token] = vector
            mx, mn = max(vector), min(vector)
            embedding_minmax[0] = embedding_minmax[0] if embedding_minmax[0] <= mn else mn
            embedding_minmax[1] = embedding_minmax[1] if embedding_minmax[0] >= mx else mx
    embedding_dim = len(list(embeddings.keys())[0])
    print("done.")
    vocab_embeddings =  dict( (w, embeddings[w]) for w in [ w for w in vocab if w in embeddings])
    missing_vocab_word_embeddings = [ w for w in vocab if w not in vocab_embeddings]
    if len(missing_vocab_word_embeddings) > 3: # UNK, EOS, BOS
        print("%d items in the vocabulary were not found in the embedding matrix!" % len(missing_vocab_word_embeddings))
        print("\n".join(missing_vocab_word_embeddings))
        error("Embedding keys error")

    else:
        for w in missing_vocab_word_embeddings:

            vocab_embeddings[w] = np.random.uniform(low=embedding_minmax[0], high=embedding_minmax[1], size=(embedding_dim,))
    # write out
    filename = vocabulary_file + ".embeddings"
    print("Writing embeddings for vocabulary at ",filename)
    mmx = [ 1000, -1000 ]
    with open(filename,"w") as fp:
        for token,vector in vocab_embeddings.items():
            mx, mn = np.max(vector), np.min(vector)
            if mx > mmx[1]: mmx[1] = mx
            if mn < mmx[0]: mmx[0] = mn
            vec = " ".join([ "%5.5f" % v for v in vector])
            fp.write("%s\t%s\n" % (token ,vec))
    print("Min / max embedding values for vocabulary: ",mmx)


else:
    error("Unsupported embeddings file type: ", embeddings_file_type)