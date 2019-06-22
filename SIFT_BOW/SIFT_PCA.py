import cv2
import os
import sys
import h5py
import numpy as np
from tqdm import tqdm
from scipy.cluster.vq import *
from sklearn.cluster import MiniBatchKMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA

if len(sys.argv) != 5:
    print ("Usage: python extract_SIFT.py image_list save_hdf5 en_PCA en_whiten")

image_list = sys.argv[1]
save_file = sys.argv[2]
en_PCA = sys.argv[3]
en_whiten = sys.argv[4]
test_list = sys.argv[5]

PCA_dim = 128
out_dim = 128
sample = True

f = open(image_list,'r')
h5 = h5py.File(save_file,'w')

all_paths = [path.strip() for path in f.readlines()]
SIFT = cv2.xfeatures2d.SIFT_create()

def get_train_path(all_paths):
    f = open(test_list,'r')
    paths = []
    for line in f.readlines():
        paths.append(line.strip())
    train_paths = []
    for path in all_paths:
        if path not in paths:
            train_paths.append(path)
    return train_paths

def get_sift(all_paths):
    
    all_list = []
    valid_paths = []
    for img_path in tqdm(all_paths):
        img = cv2.imread(img_path)
        try:
            gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            kp,des = SIFT.detectAndCompute(gray,None)
            if des.shape[1] == 128:
                all_list.append(des)
                valid_paths.append(img_path)
        except:
            print ("image {} is destoried.".format(img_path))
    return all_list, valid_paths

all_list = get_sift(all_paths)
all_matrix = np.vstack(np.asarray(all_list))

train_paths = get_train_path(all_paths)
train_list = get_sift(train_paths)
train_matrix = np.vstack(np.asarray(train_list))
print (all_matrix.shape,train_matrix.shape)

if sample:
    n_sample = 256*6000
    np.random.seed(1024)
    sample_indices = np.random.choice(all_matrix.shape[0], n_sample)
    all_matrix = all_matrix[sample_indices]

print (all_matrix.shape)

# TODO only compute training data PCA

# train PCA
if en_PCA:
    train_matrix = preprocessing.normalize(train_matrix, copy=False, norm='l2')
    pca = PCA(PCA_dim, whiten=en_whiten, copy=False).fit(train_matrix)
    all_matrix = pca.transform(all_matrix)

##### start KMeans ########
kmeans = MiniBatchKMeans(n_clusters=out_dim, random_state=0, batch_size=128)
print ("Start to cluster......")
kmeans.fit(train_matrix)
centers = kmeans.cluster_centers_

##### start quantization #######
im_features = np.zeros((len(all_paths),PCA_dim),'float32')

print (im_features.shape)
def des2feature(des, cluster_centers):
    feature = np.zeros((out_dim),'float32')
    words, distance = vq(des,cluster_centers)
    for w in words:
        feature[w] += 1
    return feature

for i in range(len(all_list)):
    des = all_list[i]
    im_features[i] = des2feature(des,centers)

im_features = preprocessing.normalize(im_features,norm='l2')

for i in range(len(valid_paths)):
    _id = os.path.basename(valid_paths[i]).split('.')[0]
    h5[_id] = np.array(im_features[i])
