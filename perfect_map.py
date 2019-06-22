import sys,pdb
import numpy as np
import csv

pred_file = sys.argv[1]

def parse_result():
    ### get predictions
    predict = {}
    f = open(pred_file,'r')
    for line in f.readlines():
       item = line.strip().split(' ')
       vidname = item[0]
       pred = item[1:]
       predict[vidname] = pred
    f.close()

    ### get groundtruth
    gt = {}
    csvfile = open('val_2019.csv','r')
    csvreader = csv.reader(csvfile)
    vid_gt = [line[:2] for line in csvreader]
    vid_gt = vid_gt[1:]
    for item in vid_gt:
        gt[item[0]] = item[1:]

    # _dict = {key: (pred,gt)}
    _dict = {}
    for key in predict.keys():
        _dict[key] = (predict[key],gt[key])

    return _dict

def calculate_ap(pred, gt):
    tp = 0
    res = []
    for i in range(len(pred)):
        if pred[i] in gt:
            tp += 1
            precision = tp*1.0/(i+1)
        else:
            precision = 0.0
        res.append(precision)
    res = np.array(res)
    ap = np.sum(res)/len(gt)
    return ap

# sum of average precisions
sum_ap = 0.0

_dict = parse_result()
keys = _dict.keys()

recordf = open(pred_file.split('_')[0]+'_records.lst','w')
for key in keys:
    pred = _dict[key][0]
    gt = _dict[key][1]

    ap = calculate_ap(pred, gt)
    recordf.write(key+' '+str(ap)+'\n')
    sum_ap += ap
print ("mAP for %s: %.5f"%(pred_file, sum_ap/len(keys)))
recordf.close()
