import random
import numpy as np
from utils.geotif_io import readTiff

def read_scene_pair(paths_scene, paths_truth):
    '''read data from path and 0-1 normalization
    '''
    scenes = []
    truths = []
    paths_scene_pair = zip(paths_scene, paths_truth)
    for path_scene, path_truth in paths_scene_pair:
        scene,_ = readTiff(path_scene)
        truth,_ = readTiff(path_truth)
        scene = np.clip(scene/10000,0,1)  # normalization
        scenes.append(scene)
        truths.append(truth)
    return scenes, truths

def crop_patch(img, truth, width=512, height=512, _random=True):
    '''crop image to patch'''
    if _random:
        row_start = random.randint(0, img.shape[0]-height)
        col_start = random.randint(0, img.shape[1]-width)
    else:
        row_start = 0
        col_start = 0
    patch = img[row_start:row_start+height, col_start:col_start+width]
    truth = truth[row_start:row_start+height, col_start:col_start+width]
    patch, truth = patch.astype(np.float32), truth.astype(np.float32)
    return patch, truth

def crop_patches(imgs, truths, width=512, height=512, _random=True):
    '''crop images to patchs
    augs:
        imgs: a list contains images
        truths: a list contains image truths
    return:
        cropted patches list truth list 
    '''
    patches, ptruths = [],[]
    for i in range(len(imgs)):
        patch, truth = crop_patch(img=imgs[i], truth=truths[i], \
                                width=width, height=height, _random=_random)
        patches.append(patch)
        ptruths.append(truth)
    return patches, ptruths

