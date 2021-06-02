from utils.tiff_io import readTiff
from dataloader.helper import crop_patch, image_aug

def read_preprocess(paths_scene, paths_truth, patch_size):
    '''read data from path
       preprocess: 
            1. 0-1 normalization; 
            2. crop scene to patch;
            3. data augmentation
    '''
    patches_pair = []
    paths_scene_pair = zip(paths_scene, paths_truth)
    for path_scene, path_truth in paths_scene_pair:
        _, scene = readTiff(path_scene)
        _, truth = readTiff(path_truth)
        scene = scene/10000   # normalization
        patch, truth = crop_patch(scene, truth, patch_size, patch_size)  # crop scene to patch
        patch_aug, truth_aug = image_aug(patch, truth, 
                                    flip = True, rot = True, noisy = True)
        patches_pair.append((patch_aug, truth_aug))

    return patches_pair

