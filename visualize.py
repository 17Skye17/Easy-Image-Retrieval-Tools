import os
import sys
from matplotlib import pyplot as plt
from tqdm import tqdm

_file = sys.argv[1]
basepath = sys.argv[2]

f = open(_file,'r')
all_paths = f.readlines()

for line in tqdm(all_paths):
    items = line.strip().split(' ')
    query = items[0]
    images = items[1:]

    plt.figure(figsize=(10,20))
    plt.subplot(432),plt.imshow(plt.imread(os.path.join(basepath,query))),plt.title('query_image')
    for i in range(len(images)):
        plt.subplot(4,3,i+4),plt.imshow(plt.imread(os.path.join(basepath,images[i])))
    plt.savefig(query)
