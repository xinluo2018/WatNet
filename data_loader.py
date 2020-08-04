#!/usr/bin/env python
# coding: utf-8

# In[11]:


# 挂载google drive
from google.colab import drive
drive.mount('/content/drive/')
# 切换工作路径
import os
os.chdir("/content/drive/My Drive/Colab/WaterMapping/Github_upload")
# !ls
# !nvidia-smi


# In[12]:


try:
    # %tensorflow_version only exists in Colab.
    get_ipython().magic(u'tensorflow_version 2.x')
except Exception:
    pass
import tensorflow as tf
import numpy as np
import random
import pathlib
from utils import readTiff


# In[13]:


folder_TrainScenes = '/content/drive/My Drive/Colab/WaterMapping/TrainingData/TrainingScene/' 
folder_TrainTruths = '/content/drive/My Drive/Colab/WaterMapping/TrainingData/TrainingTruth/'
PATCH_SIZE = 512
BATCH_SIZE = 4
BUFFER_SIZE = 200


# In[14]:


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
    return Scenes, Truths, Radios

#### Data augmentation: noisy, filp, rotate, data missing. 
def Image_aug(scene, truth, flip = True, rot = True, noisy = True, missing = True):
    scene_aug = np.copy(scene)
    truth_aug = np.copy(truth)
    if noisy == True:
        if np.random.uniform(()) > 0.5:  
            noise = np.random.normal(loc = 0, scale = 0.01, size = (512, 512, scene.shape[2]))
            scene_aug = scene_aug + noise            
    if missing == True:
        if np.random.uniform(()) > 0.75: 
            missing_wigth_row = random.randint(0,10)
            missing_wigth_col = random.randint(0,10)
            row_start = random.randint(0,scene_aug.shape[0]-missing_wigth_row)
            col_start = random.randint(0,scene_aug.shape[1]-missing_wigth_col)    
            scene_aug[row_start:row_start+missing_wigth_row, :, :] = 0
            scene_aug[:, col_start:col_start+missing_wigth_col, :] = 0
    if flip == True:
        if np.random.uniform(()) > 0.5:
            if random.randint(1,2) == 1:  ## horizontal or vertical mirroring
                scene_aug = np.flip(scene_aug, 1)
                truth_aug = np.flip(truth_aug, 1)
            else: 
                scene_aug = np.flip(scene_aug, 0)
                truth_aug = np.flip(truth_aug, 0)
    if rot == True:
        if np.random.uniform(()) > 0.5:  
            degree = random.randint(1,3)
            scene_aug = np.rot90(scene_aug, k=degree)
            truth_aug = np.rot90(truth_aug, k=degree)
    return scene_aug.astype(np.float32), truth_aug.astype(np.float32)

## crop the scenes to patches
def random_crop(input_scenes, real_scenes, radios, Patch_size):
    n_band = input_scenes[0].shape[2]
    PatchSet = []
    TruthSet = []
    for i in range(len(input_scenes)):    
        for j in range(radios[i]):
            random_row = random.randint(0,input_scenes[i].shape[0]-Patch_size)
            random_col = random.randint(0,input_scenes[i].shape[1]-Patch_size)
            stacked_scenes = np.concatenate([input_scenes[i], real_scenes[i]], axis=2).astype(np.float32)
            cropped_scenes = stacked_scenes[random_row:random_row+Patch_size, random_col:random_col+Patch_size, :]
            PatchSet.append(cropped_scenes[:,:,0:n_band])
            TruthSet.append(cropped_scenes[:,:,n_band:n_band+1])
    return PatchSet, TruthSet 

def get_scene(folder_Scenes, folder_Truths, PATCH_SIZE):
    ## input the path of the folders corresponding to scenes and truth
    path_Scenes, path_Truths = get_path(folder_Scenes, folder_Truths)
    Scenes, Truths, Ratios = load_scene(path_Scenes, path_Truths, PATCH_SIZE)
    Scenes = [np.clip(Scenes/10000, 0, 1) for Scenes in Scenes]  #   Normalization
    return Scenes, Truths

def get_patch(Scenes, Truths, BATCH_SIZE, BUFFER_SIZE):
    ## input Scenes and Truths are the list datatype
    ## return tf.data.Dataset
    Crop_Ratios = np.ones(len(Scenes), dtype=np.int)
    Patches, PatchTruths = random_crop(Scenes, Truths, 
                            radios=Crop_Ratios, Patch_size=PATCH_SIZE)
    # data augmentation
    Patches_aug = np.zeros((len(Patches), Patches[0].shape[0],
                                Patches[0].shape[1], Patches[0].shape[2]))
    PatchTruths_aug = np.zeros((len(PatchTruths), PatchTruths[0].shape[0], 
                                PatchTruths[0].shape[1], PatchTruths[0].shape[2]))
    for i in range(len(Patches)):
        Patches_aug[i], PatchTruths_aug[i] = Image_aug(Patches[i], PatchTruths[i], 
                            flip = True, rot = True, noisy = True, missing = False)       
    dataSet = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(Patches_aug), 
                                    tf.convert_to_tensor(PatchTruths_aug)))
    dataSet = dataSet.batch(BATCH_SIZE).shuffle(BUFFER_SIZE)
    return dataSet


# In[ ]:


# Scenes, Truths = get_scene(folder_TrainScenes, folder_TrainTruths, PATCH_SIZE)
# TrainSet = get_patch(Scenes, Truths, BATCH_SIZE, BUFFER_SIZE)
# TrainSet


# In[8]:




