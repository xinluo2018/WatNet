
try:
    get_ipython().magic(u'tensorflow_version 2.x')
except Exception:
    pass
import tensorflow as tf
import numpy as np
import random
import pathlib
from utils.utils import readTiff

###  Get the pathes (string) corresponding to image pathes, return a list
def get_path(folder_Scenes, folder_Truths):    
    path_Scenes = pathlib.Path(folder_Scenes)
    path_Truths = pathlib.Path(folder_Truths)
    Scene_paths = list(path_Scenes.glob('*'))
    Scene_paths = sorted([str(path) for path in Scene_paths])    
    Truth_paths = list(path_Truths.glob('*'))
    Truth_paths = sorted([str(path) for path in Truth_paths])
    return Scene_paths, Truth_paths

### load the scenes
def load_scene(Scene_paths, Truth_paths, Patch_size):
    Scenes = list(range(len(Scene_paths)))   ## initialized the list
    Truths = list(range(len(Scene_paths)))
    Radios = list(range(len(Scene_paths)))  
    for i in range(len(Scene_paths)):
        Scenes[i], _, _, im_row,im_col, _ = readTiff(Scene_paths[i])
        Truths[i], _, _, _, _, _ = readTiff(Truth_paths[i])
        Truths[i] = np.expand_dims(Truths[i], axis=2)
        Radios[i] = (im_row//Patch_size+1)*(im_col//Patch_size+1)
    ## the Radios = Scenes area/Patch area
    return Scenes, Truths, Radios

#### Data augmentation: noisy, filp, rotate. 
def image_aug(image, truth, flip = True, rot = True, noisy = True):
    if flip == True:
        if tf.random.uniform(()) > 0.5:
            if random.randint(1,2) == 1:  ## horizontal or vertical mirroring
                image = tf.image.flip_left_right(image)
                truth = tf.image.flip_left_right(truth)
            else: 
                image = tf.image.flip_up_down(image)
                truth = tf.image.flip_up_down(truth)
    if rot == True:
        if tf.random.uniform(()) > 0.5: 
            degree = random.randint(1,3)
            image = tf.image.rot90(image, k=degree)
            truth = tf.image.rot90(truth, k=degree)
    if noisy == True:
        if tf.random.uniform(()) > 0.5:
            std = random.uniform(0.002, 0.02)
            gnoise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=std, dtype=tf.float32)
            image = tf.add(image, gnoise)
    return image, truth

# ## crop the scenes to patches
# def random_crop(input_scenes, real_scenes, radios, Patch_size):
#     n_band = input_scenes[0].shape[2]
#     # PatchSet = []
#     # TruthSet = []
#     for i in range(len(input_scenes)):    
#         for j in range(radios[i]):
#             random_row = random.randint(0,input_scenes[i].shape[0]-Patch_size)
#             random_col = random.randint(0,input_scenes[i].shape[1]-Patch_size)
#             stacked_scenes = np.concatenate([input_scenes[i], real_scenes[i]], axis=2).astype(np.float32)
#             cropped_scenes = stacked_scenes[random_row:random_row+Patch_size, random_col:random_col+Patch_size, :]
#             # PatchSet.append(cropped_scenes[:,:,0:n_band])
#             # TruthSet.append(cropped_scenes[:,:,n_band:n_band+1])
#     return cropped_scenes[:,:,0:n_band], cropped_scenes[:,:,n_band:n_band+1]

def get_scene(folder_Scenes, folder_Truths, PATCH_SIZE):
    ## input the path of the folders corresponding to scenes and truth
    path_Scenes, path_Truths = get_path(folder_Scenes, folder_Truths)
    Scenes, Truths, Ratios = load_scene(path_Scenes, path_Truths, PATCH_SIZE)
    Scenes = [np.clip(Scenes/10000, 0, 1) for Scenes in Scenes]  #   Normalization
    return Scenes, Truths

def get_patch(Scenes, Truths, PATCH_SIZE, BATCH_SIZE, BUFFER_SIZE):
    ## input Scenes and Truths are the list datatype
    ## return tf.data.Dataset
    # Crop_Ratios = np.ones(len(Scenes), dtype=np.int)
    # Patches, PatchTruths = random_crop(Scenes, Truths, 
    #                         radios=Crop_Ratios, Patch_size=PATCH_SIZE)
    # data augmentation
    stacked = list(zip(Scenes,Truths))
    stacked = [np.concatenate(imgPair,axis=2) for imgPair in stacked]
    stacked = [tf.convert_to_tensor(imgPair, dtype=tf.float32) for imgPair in stacked]
    # cropped_stacked = tf.image.random_crop(
    #                     stacked, size=[512, 512, stacked.shape[2]])    
    # Patches, PatchTruths = cropped_stacked[:,:,:stacked.shape[2]], cropped_stacked[:,:,stacked.shape[2]:] 
    # Patches_aug = np.zeros((len(Patches), Patches[0].shape[0],
    #                             Patches[0].shape[1], Patches[0].shape[2]))
    # PatchTruths_aug = np.zeros((len(PatchTruths), PatchTruths[0].shape[0], 
    #                             PatchTruths[0].shape[1], PatchTruths[0].shape[2]))
    # Patches_aug = np.zeros((len(Scenes), PATCH_SIZE, PATCH_SIZE, Scenes[0].shape[2]))
    # PatchTruths_aug = np.zeros((len(Scenes), PATCH_SIZE, PATCH_SIZE, Truths[0].shape[2]))
    Patches_aug = []
    PatchTruths_aug = []
    # for i in range(len(Patches)):
    #     Patches_aug[i], PatchTruths_aug[i] = Image_aug(Patches[i], PatchTruths[i], 
    #                         flip = True, rot = True, noisy = True, missing = False)       
    for imgPair in stacked:
        imgPair = tf.image.random_crop(imgPair, size=[512, 512, imgPair.shape[2]])    
        Patch, PatchTruth = imgPair[:,:,:Scenes[0].shape[2]], imgPair[:,:,Scenes[0].shape[2]:]
        Patch_aug, PatchTruth_aug = image_aug(Patch, PatchTruth,\
                                 flip = True, rot = True, noisy = True)
        Patches_aug.append(Patch_aug)
        PatchTruths_aug.append(PatchTruth_aug)
    # dataSet = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(Patches_aug),\
    #                                 tf.convert_to_tensor(PatchTruths_aug)))
    dataSet = tf.data.Dataset.from_tensor_slices((Patches_aug,PatchTruths_aug))
    dataSet = dataSet.batch(BATCH_SIZE).shuffle(BUFFER_SIZE)
    return dataSet