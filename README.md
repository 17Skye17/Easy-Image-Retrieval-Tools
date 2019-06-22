# An Easy Image Retrieval Toolkit

## Features

+ Extract SIFT descriptor

+ Extract popular CNN features: inceptionresnetv2/inceptionv4/resnet/resnext/xception/senet/polynet etc. see `pretrainedmodels/models` for more details.

+ Feature Postprocess: PCA + whiten

+ Visualize top k image retrieval results.

## Requirements

```shell
pytorch >=1.0
torchvision
PIL
h5py
numpy
tqdm
munch
matplotlib
```

## Usage:

### 1. Generate image lists
```shell
python gen_datalist.py img_dir image_list 
```

example:

```shell
python gen_datalist.py  dataset/paris6k/train train.lst
```
Same to `val.lst`

### 2. Extract features
```shell
python extract.py model_name image_list save_file gpu_id batch_size
```

example:

```shell
python extract.py resnet101 train.lst paris-train.hdf5 0 128
```
Same to `val.lst`

### 3. Evaluate

#### For `paris6k, oxford5k, roxford5k, rparis6k`, use `evaluate.py`

Note: you can modify `utils/parser.py` to fit your need
```shell
python evaluate.py --datasets=paris6k --features=paris-val.hdf5
```
#### For your own dataset:

##### 1.Calculate distance

```shell
cd eval/

# calculate distance
python hnsw.py train_db test_db top_k
```

example:

```shell
python hnsw.py paris-train.hdf5 paris-val.hdf5 5
```
A file named `paris_rank` is generated, format:
```shell
val_image_id train_image_id1 train_image_id2 ... train_image_idk
```
##### 2.Caltulate mAP

Use **mAP** as evaluation metric:
```shell
python map.py pred_file
```

example:

```shell
python map.py paris_rank
```
I tested this code on ThePerfect500k dataset, please verify the ground truth file in `map.py` which is a csv file in format:

```shell
val_image_id, train_image_id1 ... train_image_idn
```


Update by Skye

06/22/2019
