from fastai.vision import *
import argparse

# init argument parser
parse = argparse.ArgumentParser()
parse.add_argument('-i', '--image', help='path to image to run inference on')
args = vars(parse.parse_args())

# load dataset to get codes
path = untar_data(URLs.CAMVID)
codes = np.loadtxt(path/'codes.txt', dtype=str)

# Load metric functions for inference
name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']

def acc(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

# load inference model
inference = load_learner('output/', 'model.pkl')

# load image
img = open_image(args['image'])

# show result
img.show(y=inference.predict(img)[0])