import numpy as np
from utils_ import *
from process_annotations import read_vocabulary

'''
Script to encode vocabulary contents into the embedding matrix. 
'''
# vocabulary and embeddings files
vocabulary_file = "/home/nik/uoa/msc-thesis/implementation/vocabulary_captions_train2014.json.vocab"
embeddings_file = "/home/nik/uoa/msc-thesis/dataset/glove/glove.6B.50d.txt"
embedding_dim = 50
# the range of embedding values
embedding_value_minmax =  [-3.0575000000000001, 4.3657000000000004]
embeddings_file_type = "glove"


vocab = read_vocabulary(vocabulary_file)

omit_count = 0
total_count = 0
if embeddings_file_type == "glove":
    embeddings = {}
    # read the whole contents
    print("Reading embeddings file...",end="")
    with open(embeddings_file,"r") as fp:
        for line in fp:
            contents = line.split()
            token = contents[0]
            vector = list(map(float, contents[1:]))
            embeddings[token] = vector
    print("done.")
    vocab_embeddings =  dict( (w, embeddings[w]) for w in [ w for w in vocab if w in embeddings])
    missing_vocab_word_embeddings = [ w for w in vocab if w not in vocab_embeddings]
    if len(missing_vocab_word_embeddings) > 3: # UNK, EOS, BOS
        print("%d items in the vocabulary were not found in the embedding matrix!" % len(missing_vocab_word_embeddings))
        print("\n".join(missing_vocab_word_embeddings))
        error("Embedding keys error")

    else:
        for w in missing_vocab_word_embeddings:

            vocab_embeddings[w] = np.random.uniform(low=embedding_value_minmax[0], high=embedding_value_minmax[1], size=(embedding_dim,))
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
    print("Min / max embedding values : ",mmx)


else:
    error("Unsupported embeddings file type: ", embeddings_file_type)