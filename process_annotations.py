import json, string, os, sys
from utils_ import  init_config

# do initializations
init_file = "config.ini"
if len(sys.argv) > 1:
    init_file = sys.argv[1]
print("Using initialization file : ", init_file)

#########################
# settable parameters
# caption files to process
caption_files = ["/home/nik/uoa/msc-thesis/dataset/coco14/annotations/captions_val2014.json"]
# format per file
caption_file_formats = ("coco", "flickr")
# vocabulary file: if not None, it will produce caption encodings as per the vocabulary
vocabulary_file = None
# replacement file: a file containing w, [v1,v2,..]. Each occurence of w in a caption will be replaced by v1,v2,...]
# this is so that weird slang and compositions are replaced by common words, for which pretrained embeddings exist
vocab_replacement_file = "/home/nik/uoa/msc-thesis/dataset/glove/missing_words.txt"
caption_max_length = 16
word_count_thresh = 5
########################


if __name__ == '__main__':
    keyvals = init_config(init_file, "captions")

    for key in keyvals:
        exec("%s=%s" % (key, keyvals[key]), )
    print("Successfully initialized from file %s" % init_file)


# function that replaces tokens with replacements, so as to match tokens in embeddings
def replace_problematic_words(toklist, replacements):

    for w in replacements:
        for i,t in enumerate(toklist):
            if t == w:
                toklist = [*toklist[:i] , * replacements[w].split(), *toklist[i+1 :] ]
                # print("Replaced %s with %s" % (w,toklist[i : i + len(replacements[w].split())]))
    return toklist

# read caption data from file
def read_file(filename, format):
    print("Reading file ",filename)
    img_captions = None
    print("File format is %s." % format)

    # read Coco caption data
    if format == "coco":
        with open(filename,'r') as f:
            print("Loading json data.")
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
    return image_jsons

# preprocess the captions, removing punctuation and applying token replacement if needed
def prepro_captions(imgs_json, vocab_replacement_file):

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
                tokens = line.strip().split("\t")
                tokens = [ tok.strip() for tok in tokens if len(tok.strip()) > 0]
                word = tokens[0]
                replacements[word] = " ".join(tokens[1:])

        for i, img in enumerate(imgs_json):
            for t, txt in enumerate(img['processed_tokens']):
                txt = replace_problematic_words(txt, replacements)
                imgs_json[i]['processed_tokens'][t] = txt

# construct the vocabulary from the input captions
def build_vocab(imgs, word_count_thresh):
    if word_count_thresh is not None:
        vocab = apply_frequency_filtering(imgs, word_count_thresh)
    else:
        vocab = set()
        for img in imgs:
            for txt in img['processed_tokens']:
                for w in txt:
                    vocab.add(w)
        vocab = list(vocab)
    return vocab

# produce a frequenccy-filtered vocabulary
def apply_frequency_filtering(imgs, count_threshold ):
    print("Counting tokens...")
    counts = {}
    for img in imgs:
        for txt in img['processed_tokens']:
            for w in txt:
                if w not in counts:
                    counts[w] = 1
                else:
                    counts[w] = counts[w] + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)

    # print top words
    num_top = 10
    print('\ntop %d words and their counts:' % num_top)
    print('\n'.join(map(str, cw[:num_top])))
    print()

    vocab = [w for w, n in counts.items() if n > count_threshold]

    # print some stats
    total_words = len(counts.items())
    bad_words = [w for w, n in counts.items() if n <= count_threshold]

    print('total words:', total_words)
    print('number of non-frequent words to-be-mapped to UNK: %d/%d = %.2f%%' % (
        len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab: %d' % (len(vocab)))
    return vocab

# map captions to vocabulary tokens
def finalize_captions(img_list, captions_field, vocab, caption_max_length):
    for img in img_list:
        img['final_captions'] = []
        # map to vocabulary and potentially truncate
        for raw_word_list in img[captions_field]:
            vocab_words = [w if w in vocab else 'UNK' for w in raw_word_list]
            if caption_max_length is not None and len(vocab_words) > caption_max_length:
                trunc_vocab_words= vocab_words[0:caption_max_length]
                print("Limiting caption of size %d :" % len(vocab_words), str(vocab_words), " to ", str(trunc_vocab_words))
                vocab_words = trunc_vocab_words
            img['final_captions'].append(vocab_words)

# read the vocab
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


def main():

    # read caption files
    print (caption_file_formats)
    image_jsons = []
    for i,c in enumerate(caption_files):
        image_jsons.append(read_file(c, caption_file_formats[i]))

    for i,c in enumerate(image_jsons):
        print('Processing tokens of %s.' % (caption_files[i]))
        prepro_captions(c, vocab_replacement_file)

    # if no vocabulary specified, build it
    if vocabulary_file is None:
        img_json = []
        for obj in image_jsons:
            img_json.extend(obj)

        vocab = build_vocab(img_json, word_count_thresh)
        # add EOS, BOS
        vocab.extend(["UNK","EOS","BOS"])
        # write to the folder of the first caption file
        filename =  os.path.join(os.path.dirname(caption_files[0]),
                                 "_".join([ os.path.basename(capfile) for capfile in caption_files]))
        filename = filename + ".vocab"
        print("Produced vocabulary of", len(vocab)," words, including the UNK, EOS, BOS symbols.")
        print("Writing vocabulary to",filename )
        with open(filename, "w") as f:
            for w in vocab:
                f.write(w + "\n")
    else:
        # encode captions to an existing vocabulary
        vocab = read_vocabulary(vocabulary_file)
        for i in range(len(caption_files)):
            filename = caption_files[i]
            imgjson = image_jsons[i]

            print("Mapping captions of ", filename, " to the vocabulary")
            finalize_captions(imgjson,"processed_tokens",vocab, caption_max_length)

            # write
            with open(filename + ".paths.txt","w") as f:
                for image_obj in imgjson:
                    imgname = image_obj["filename"]
                    for cap in image_obj['final_captions']:
                        labels = []
                        for word in cap:
                            if word not in vocab:
                                print("Word",word,"not found in vocabulary.")
                                exit(1)
                            labels.append(str(vocab[word]))
                        f.write("%s %s\n" % (imgname, " ".join(labels)))
            print("Wrote file ",filename + ".paths.txt")


if __name__ == "__main__":
   main()
