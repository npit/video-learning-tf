from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import os, sys, getopt

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')


results_dir = 'best_model/results'

def evaluateModel(model_json, coco):
    cocoRes = coco.loadRes(model_json)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()  
    cocoEval.evaluate()
    results = {}
    for metric, score in cocoEval.eval.items():
        results[metric] = score
    return results

def main(argv):
    if len(argv) < 2:
    	print 'Missing results files argument'
    	exit (1)
    resFile = argv[0]
    gtFile = argv[1]
    print "\n\n"
    print 'Evaluating imageids - caption file :file %s' % resFile
    print 'Using ground truth file %s' % gtFile

    coco = COCO(gtFile)
    results = evaluateModel(resFile, coco)
    final_eval_file = resFile + '.eval.json'
    print 'Writing scores to file %s' % final_eval_file

    if not os.path.isfile(final_eval_file):
        all_results_json = {}
    else:
        with open(final_eval_file,'r') as f:
            all_results_json = json.load(f)

    with open(final_eval_file,'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)

if __name__ == '__main__':
    main(sys.argv[1:])
