import numpy as np
from utils_ import *
from process_annotations import read_vocabulary

'''
Script to encode a vocabulary into the embedding matrix. 
'''

vocabulary_file = "/home/nik/uoa/msc-thesis/implementation/vocabularyresults_20130124.token.vocab"
embeddings_file = "/home/nik/uoa/msc-thesis/dataset/glove/glove.6B.50d.txt"
embedding_dim = 50
embeddings_file_type = "glove"
vocab_replacement_file="/home/nik/uoa/msc-thesis/dataset/glove/missing_words.txt"
vocab_replacement_file= None
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
    if len(missing_vocab_word_embeddings) > 3:
        print("%d items in the vocabulary were not found in the embedding matrix!")
        # for single word replacements, we can assign the embedding of the replacement word for the missing word
        # i.e. we can assign the embedding for the existing word 'hoop' to the missing hulahoop

        # for replacements like redhaired -> red haired, we have to change the caption.
        # we should probably do this for all cases, for consistency.

        # print("\n".join(missing_embeddings))
        # attemp to fix them
        if not vocab_replacement_file:
            error("Vocabulary - embedding errors with no replacement file")

        replacements = {}
        with open(vocab_replacement_file,"r") as f:
            for line in f:
                word,repl = line.strip().split("\t")
                repl = repl.split()
                replacements[word] = repl

        replaced = [replacements[w] for w in missing_vocab_word_embeddings]
        all_words_replacements = []
        for w in replaced:
            all_words_replacements.extend(w)
        for w in all_words_replacements:
            if w not in vocab:
                print("Adding replacement word %s in vocab" % w)
                vocab[w] = len(vocab)
        still_missing = [w for w in all_words_replacements if w not in embeddings]
        if len(still_missing) > 3:
            error("Unfixable missing embeddings.")
        # encode the special characters
        for w in still_missing:
            vocab_embeddings[w] = np.random.rand(embedding_dim)
    else:
        minmax = [-3.0575000000000001, 4.3657000000000004]
        for w in missing_vocab_word_embeddings:

            vocab_embeddings[w] = np.random.uniform(low=minmax[0], high=minmax[1], size=(embedding_dim,))
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
    print(mmx)


else:
    error("Unsupported embeddings file type: ", embeddings_file_type)