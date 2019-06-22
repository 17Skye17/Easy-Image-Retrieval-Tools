import pretrainedmodels
import sys
import torch
import time
import h5py
import os
import numpy as np
import torchvision
from DataReader import ImageDataset,TransformImage
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

if len(sys.argv)!=6:
    print ("Usage: python extract.py model_name image_list save_file gpu_id")

model_name = sys.argv[1]
image_list = sys.argv[2]
db_file = sys.argv[3]
gpu_id = sys.argv[4]
batch_size = sys.argv[5]

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

Transformer = TransformImage(model)

Image = ImageDataset(image_list, transform=Transformer)

image_loader = torch.utils.data.DataLoader(
            dataset=Image, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True)

image_id = open('image_id.lst','w')

feat_db = h5py.File(db_file, 'w')
with torch.no_grad():
    
    model.eval()
    
    for batch_idx, (img_path, data) in enumerate(image_loader):
        st = time.time()
        data = data.cuda()
        pool_feat = model.pool_feat(data)
        pool_feat = pool_feat.squeeze() 
        print (pool_feat.size())

        std = time.time()
        print ("processed {} videos, time elapse: {}s/batch".format(batch_idx*batch_size,(std-st)))
        for _imgpath, _pool in zip(img_path, pool_feat):
            feat_db[os.path.basename(_imgpath).split('.')[0]] = _pool.cpu().numpy()
