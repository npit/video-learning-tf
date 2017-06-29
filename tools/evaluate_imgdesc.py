from tools.coco_eval.pycocotools.coco import COCO
from tools.coco_eval.pycocoevalcap.eval import COCOEvalCap

import os, sys, getopt

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')


def evaluate_coco(results_json_file, ground_truth_file):
    coco = COCO(ground_truth_file)

    cocoRes = coco.loadRes(results_json_file)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    results = {}
    for metric, score in cocoEval.eval.items():
        results[metric] = score
    return results
