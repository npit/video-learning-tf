from sklearn.metrics import f1_score, precision_score, recall_score
import argparse
import pickle
import numpy

path_dset="/run/media/nik/TOSHIBA EXT/Academic/msc-thesis/datasets/kth/serialized/val.txt"
path_logits="/run/media/nik/TOSHIBA EXT/Academic/msc-thesis/runs/kth/singleframe/validation_logits_config.kth.vec.yml_val_resume_120618_175935.total"


labels=[]
logits=None
with open(path_dset) as f:
    for line in f:
        lbl = int(line.strip().split()[-1])
        labels.append(lbl)

with open(path_logits,"rb") as f:
    logits=pickle.load(f)

amax=numpy.argmax(logits, axis=1)
print("Logits:", path_logits)
print("F1 PREC REC")
print(f1_score(labels, amax.tolist(),average='macro'))
print(precision_score(labels, amax.tolist(),average='macro'))
print(recall_score(labels, amax.tolist(),average='macro'))
