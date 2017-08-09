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
randomize_missing_embeddings = False

if len(sys.argv) > 1:
    init_file = sys.argv[1]
print("Using initialization file : ",init_file)

keyvals = init_config(init_file, "captions")

for key in keyvals:
    exec("%s=%s" % (key, keyvals[key]))
print("Successfully initialized from file %s" % init_file)

# read the vocabulary
vocab = read_vocabulary(vocabulary_file)

# switch by type of embeddings read
if embeddings_file_type == "glove":
    embeddings = {}

    # read the whole contents and compute the min/max in which to produce random embeddings for not found words
    print("Reading embeddings file...",end="")
    with open(embeddings_file,"r") as fp:
        for line in fp:
            contents = line.split()
            token = contents[0]
            vector = list(map(float, contents[1:]))
            embeddings[token] = vector

    keys = list(embeddings.keys())
    key = keys[0]
    embedding_dim = len(embeddings[key])

    just_embeddings = [embeddings[w] for w in embeddings]
    embedding_minmax = [np.amin(just_embeddings), np.amax(just_embeddings)]
    print("Embedding dimension read :", embedding_dim, "\nDone.")
    print("Embedding min/max read :", str(embedding_minmax))

    vocab_embeddings =  dict( (w, embeddings[w]) for w in [ w for w in vocab if w in embeddings])
    missing_vocab_word_embeddings = [ w for w in vocab if w not in vocab_embeddings]
    if len(missing_vocab_word_embeddings) > 3: # apart from EOS, BOS, UNK
        print("%d items in the vocabulary were not found in the embedding matrix (other than EOS,BOS)!" %
              (3-len(missing_vocab_word_embeddings)))
        print("\n".join(missing_vocab_word_embeddings))
    else:
        print("No missing embeddings other than EOS, BOS and UNK.")
    if not randomize_missing_embeddings:
        print("Randomizing embeddings is disabled, exiting.")
        exit(1)

    for w in missing_vocab_word_embeddings:
        arr = np.random.uniform(low=embedding_minmax[0], high=embedding_minmax[1], size=(embedding_dim,))
        print("Producing vector for missing token:",w)
        vocab_embeddings[w] = arr

            
    # write out
    filename = vocabulary_file + ".embeddings"
    print("Writing embeddings for vocabulary at ",filename)

    with open(filename,"w") as fp:
        for token,vector in vocab_embeddings.items():
            vec = " ".join([ "%5.5f" % v for v in vector])
            fp.write("%s\t%s\n" % (token ,vec))
else:
    error("Unsupported embeddings file type: ", embeddings_file_type)
