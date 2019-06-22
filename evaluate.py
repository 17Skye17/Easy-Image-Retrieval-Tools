import time
import os
import h5py
import numpy as np
from tqdm import tqdm
from utils.parser import get_parse
from utils.testdataset import configdataset
from utils.general import get_data_root, htime
from utils.evaluate import compute_map_and_print

args = get_parse()

datasets = args.datasets.split(',')

h5 = h5py.File(args.features,'r')
keys = list(h5.keys())

def get_features(images):
    vecs = []
    for img in tqdm(images):
        basename = os.path.basename(img).split('.')[0]
        vecs.append(np.array(h5[basename]))
    vecs = np.asarray(vecs)
    return vecs

for dataset in datasets:
    start = time.time()
    cfg = configdataset(dataset, os.path.join(get_data_root(),'test'))
    images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
    qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
    bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]

    print('{}: database images...'.format(dataset))

    vecs = get_features(images)
    qvecs = get_features(qimages)
    print (vecs.shape, qvecs.shape)    
    
    scores = np.dot(qvecs, vecs.T)
    print ("scores shape = {}".format(scores.shape))
    ranks = np.argsort(-scores,axis=1)
    compute_map_and_print(dataset, ranks, cfg['gnd'])

    print ('{}: elapsed time : {}'.format(dataset, htime(time.time()-start)))
