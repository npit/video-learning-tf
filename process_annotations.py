import json

# todo: add this as another mode is serialize?
# produce a frames list file out of all train, val images, as in act. rec. , associate each path with .. what? label to unique caption?
# dunno. maybe:
# associate all words with a onehot
#       -- ALL words? what about junk words? => but junk words are indeed in captions
#       -- prolly all words
# thus associate each img with label series, corresponding to the sequence of onehot vecs (words)
# seqlen is 16, so i guess captions have to be 16 long
# dont forget to add BOS and EOS symbols

# reread
caption_files = [ "/home/nik/uoa/msc-thesis/dataset/coco14/annotations/captions_val2014.json"]

for cf in caption_files:

    with open(cf,'r') as f:
        data = json.load(f)

for key in data:
    print("%s : %d " % (key, len(key)))

a =2

