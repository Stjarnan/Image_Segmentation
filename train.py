from fastai.vision import *
from fastai.callbacks.hooks import *

# Load data
path = untar_data(URLs.CAMVID)

# store label and image paths
path_label = path/'labels'
path_image = path/'images'

file_names = get_image_files(path_image)
label_names = get_image_files(path_label)

# Create label function
# get base mask info
get_label_func = lambda o: path_label/f'{o.stem}_P{o.suffix}'
img_file = file_names[0]
mask = open_mask(get_label_func(img_file))
src_size = np.array(mask.shape[1:])

# Classcodes
codes = np.loadtxt(path/'codes.txt', dtype=str)

# base hypers
SIZE = src_size // 2
BATCH_SIZE = 8

# create the src-list using fastai's segmentationitemlist
src = (SegmentationItemList.from_folder(path_image)
    .split_by_fname_file('../valid.txt')
    .label_from_func(get_label_func, classes=codes))

# init transform before use to prevent error
tfms = get_transforms()

# init datahandler
data = (src.transform(tfms, size=SIZE, tfm_y=True)
    .databunch(bs=BATCH_SIZE, num_workers=8)
    .normalize(imagenet_stats))

# functions to handle metrics
name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']

def acc(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

# create learn object
learn = unet_learner(data, models.resnet34, metrics=acc, wd=1e-2)

# train for one cycle
# unfreeze and fit for another cycle
learn.fit_one_cycle(10, slice(1e-06, 1e-03), pct_start=0.9)
learn.unfreeze()
learn.fit_one_cycle(12, slice(1e-5, 1e-4), pct_start=0.8)

# save model
learn.save('fastai-segmentation-model')