import os
import nmslib
import h5py
import time
import sys
import numpy as np
from tqdm import tqdm

if len(sys.argv) != 4:
    print ("Usage: python hnsw.py feature_db top_k")

train_db = sys.argv[1]
test_db = sys.argv[2]
topk = int(sys.argv[3])

def get_data(feature_db):
    h5 = h5py.File(feature_db,'r')
    keys = list(h5.keys())
    total = []
    video_ids = []
    for key in tqdm(keys):
        total.append(np.array(h5[key]))
        video_ids.append(key)
    video_ids = np.array(video_ids)
    total = np.array(total)
    return total,video_ids

st = time.time()
train_data,train_ids = get_data(train_db)
test_data, test_ids = get_data(test_db)

#data = np.concatenate((train_data,test_data),axis=1)

index = nmslib.init(method='hnsw', space='cosinesimil')
index.addDataPointBatch(train_data)
index.createIndex({'post': 2}, print_progress=True)

neighbours = index.knnQueryBatch(test_data,k=topk,num_threads=20)
neighbor_index = [neighbor[0] for neighbor in neighbours]
neighbor_index= np.array(neighbor_index)

f = open(os.path.basename(train_db).split('-')[0]+'_rank','w')
for j in range(len(test_ids)):
    _id = test_ids[j]
    line = _id + ' '
    topk_index = neighbor_index[j]
    for key in topk_index:
        line = line + train_ids[key] + ' '
    f.write(line+'\n')
f.close()

# query for the nearest neighbours of the first datapoint
stp = time.time()
interval = stp - st
print (interval)
