import json, string,  os
from utils_ import  init_config


caption_max_length = 16
word_count_thresh = 5

# todo: add this as another mode is serialize?
# produce a frames list file out of all train, val images, as in act. rec. , associate each path with .. what? label to unique caption?
# dunno. maybe:
# associate all words with a onehot
#       -- ALL words? what about common/stop words? => but such words are indeed in captions
#       -- prolly all words
# thus associate each img with label series, corresponding to the sequence of onehot vecs (words)
# seqlen is 16, so i guess captions have to be 16 long
# dont forget to add BOS and EOS symbols

# what about labels like 'MT' that are in there...

# preprocessing captions follows https://github.com/karpathy/neuraltalk2

# karpathy et. al caption preprocessing and vocab building



# study karpathy imgdesc papers

init_file = "config.ini"
#########################
# settable parameters
# caption files to process
caption_files =  ["/home/nik/uoa/msc-thesis/dataset/coco14/annotations/captions_val2014.json"]
# format per file
caption_file_formats = ("coco", "flickr")
# vocabulary file: if not None, it will produce caption encodings as per the vocabulary
vocabulary_file = None
# if true, it will first seek for already processed caption files before generating them
can_load_processed = False
# replacement file: a file containing w, [v1,v2,..]. Each occurence of w in a caption will be replaced by v1,v2,...]
# this is so that weird slang and compositions are replaced by common words, for which pretrained embeddings exist
vocab_replacement_file="/home/nik/uoa/msc-thesis/dataset/glove/missing_words.txt"
########################

# recognizable formats to automatically parse
avail_formats = ["coco", "flickr"]
keyvals = init_config(init_file, "captions")

for key in keyvals:
    exec("%s=%s" % (key, keyvals[key]))
print("Successfully initialized from file %s" % init_file)

#caption_files, formats, vocabulary_file, can_load, vocab_replacement_file = \
# make generic configuration reader

def replace_problematic_words(toklist, replacements):

    for w in replacements:
        for i,t in enumerate(toklist):
            if t == w:
                toklist = [*toklist[:i] , * replacements[w].split(), *toklist[i+1 :] ]
                # print("Replaced %s with %s" % (w,toklist[i : i + len(replacements[w].split())]))
    return toklist
# read data
def read_file(filename, format):
    print("Reading file ",filename)
    img_captions = None
    if not os.path.exists(filename + ".per_image.json") or ( not can_load_processed ):
        # read Coco caption data
        if format == "coco":
            print("Reading %s file." % format)
            with open(filename,'r') as f:
                print("Loading caption file : %s" % filename)
                data = json.load(f)
            print("Reading data.")
            img_captions = {}
            img_filenames = {}
            # read image ids and associate them with their captions
            for annot_item in data['annotations']:
                image_id = annot_item['image_id']
                caption = annot_item['caption']
                if not image_id in img_captions:
                    img_captions[image_id] = []
                img_captions[image_id].append(caption)
            # also read the image file name
            for image_dict in data['images']:
                image_file = image_dict['file_name']
                image_id = image_dict['id']
                img_filenames[image_id] = image_file

        # read flickr30 caption data
        elif format == "flickr":
            print("Reading %s file." % format)
            lines = []
            with open(filename,"r") as f:
                for line in f:
                    lines.append(line.strip())
            print("Reading data.")
            img_captions = {}
            img_filenames = {}
            for line in lines:
                img,caption = line.split("\t")
                name, numcaption = img.split("#")
                if name not in img_captions:
                    img_captions[name] = []
                img_captions[name].append(caption)
                img_filenames[name] = name

        # combine captions per image
        print("Generating json object.")
        image_jsons = []
        for id in img_captions:
            obj = {}
            obj["id"] = id
            obj["filename"] = img_filenames[id]
            obj["raw_captions"] = []
            for cap in img_captions[id]:
                obj["raw_captions"].append(cap)
            image_jsons.append(obj)
        with open(filename + ".per_image.json", "w") as fp:
            json.dump(image_jsons, fp)

    # else, the file exists : load it
    else:
        print("Loading existing %s file." % format)
        with open(filename + ".per_image.json","r") as fp:
            image_jsons = json.load(fp)

    return image_jsons

def prepro_captions(imgs_json):
    # preprocess all the captions

    translator = str.maketrans('', '', string.punctuation)

    for i, img in enumerate(imgs_json):
        img['processed_tokens'] = []
        for j, s in enumerate(img['raw_captions']):
            txt = str(s).lower().translate(translator).strip().split()
            img['processed_tokens'].append(txt)

    if vocab_replacement_file is not None:
        # read replacements file
        replacements = {}
        with open(vocab_replacement_file, "r") as f:
            for line in f:
                word, repl = line.strip().split("\t")
                repl = repl.split()
                replacements[word] = " ".join(repl)

        for i, img in enumerate(imgs_json):
            for t, txt in enumerate(img['processed_tokens']):
                txt = replace_problematic_words(txt, replacements)
                imgs_json[i]['processed_tokens'][t] = txt


def build_vocab(imgs):
    count_thr = word_count_thresh

    # count up the number of words

    print()

    counts = {}
    for img in imgs:
        for txt in img['processed_tokens']:
            for w in txt:
                if w not in counts:
                    counts[w] = 1
                else:
                    counts[w] = counts[w] + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('\ntop words and their counts:')
    print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = len(counts.items())
    print('total words:', total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    good_count = sum(counts[w] for w in vocab)
    total_count = sum( counts[w] for w in counts)
    print('number of bad words: %d/%d = %.2f%%' % (
    len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for txt in img['processed_tokens']:
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len + 1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('UNK')

    for img in imgs:
        img['final_captions'] = []
        for txt in img['processed_tokens']:
            caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
            if caption_max_length is not None and len(caption) > caption_max_length:
                print("Limiting caption of size %d " % len(caption),caption," to ",caption[0:caption_max_length])
                caption = caption[0:caption_max_length]
            img['final_captions'].append(caption)

    return vocab

def finalize_captions(img_list, captions_field, vocab):
    for img in img_list:
        img['final_captions'] = []
        for txt in img[captions_field]:
            caption = [w if w in vocab else 'UNK' for w in txt]
            img['final_captions'].append(caption)

def read_vocabulary(vocab_file):
    # read vocabulary
    print("Reading vocabulary from ",vocab_file)
    vocab = {}
    count = 0
    with open(vocab_file, "r") as f:
        for line in f:
            if not line or len(line) == 0:
                continue
            vocab[line.strip()] = count
            count = count + 1
    print("Read a %d-word vocabulary." % len(vocab))
    return vocab

#some confusion as to where we call the vocab() method , how we process paths when the vocab has been created..

def main():

    print("Limit caption length?")

    image_jsons = []
    for i,c in enumerate(caption_files):
        image_jsons.append(read_file(c, caption_file_formats[i]))
    for i,c in enumerate(image_jsons):
        print('Processing tokens of %s.' % (caption_files[i]))
        prepro_captions(c)





    # if train, build vocabulary
    if vocabulary_file is None:
        img_json = []
        for obj in image_jsons:
            img_json.extend(obj)

        vocab = build_vocab(img_json)
        # add EOS, BOS
        vocab.extend(["EOS","BOS"])
        filename = "vocabulary_" + "_".join([ os.path.basename(capfile) for capfile in caption_files])
        print("Produced vocabulary of", len(vocab)," words, including the UNK, EOS, BOS symbols.")
        print("Writing vocabulary to ",filename + ".vocab")
        with open(filename + ".vocab", "w") as f:
            for w in vocab:
                f.write(w + "\n")
    else:
        # finalize captions

        vocab = read_vocabulary(vocabulary_file)
            # cycle through the images, create a seqlen x vocab-len one-hot vector out of their caption
        # encode the seq of one hot to a seq of integers, write path - seq_of_ints
        for i in range(len(caption_files)):
            filename = caption_files[i]
            imgjson = image_jsons[i]

            # finalize captions
            print("Mapping captions of ", filename, " to the vocabulary")

            finalize_captions(imgjson,"processed_tokens",vocab)
            with open(filename + ".paths.txt","w") as f:
                for image_obj in imgjson:
                    imgname = image_obj["filename"]
                    for cap in image_obj['final_captions']:
                        labels = []
                        for word in cap:
                            labels.append(str(vocab[word]))
                        f.write("%s %s\n" % (imgname, " ".join(labels)))
            print("Wrote file ",filename + ".paths.txt")



if __name__ == "__main__":
   main()