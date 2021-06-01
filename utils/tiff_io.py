import rasterio as rio
import numpy as np

def readTiff(path_in):
    ''' 
        author: xin luo, date: 2021.5.6
        input: path of the .tif image
        output: rasterio DatasetReader object, and np.array data (image)
    '''
    img_src = rio.open(path_in)
    if img_src.count > 1:
        bands_ind = [i for i in range(1, img_src.count+1)]
        img_array = img_src.read(bands_ind)
        img_array = img_array.transpose(1, 2, 0)
    else: 
        img_array = img_src.read(1)
    return img_src, img_array

def writeTiff(im_data, im_transform, im_crs, path_out):
    '''im_data: 3d (row, col, band) np.array.'''
    if len(im_data.shape) > 2:
        im_data = im_data.transpose(2,0,1)
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
        im_data = np.expand_dims(im_data, axis=0)
    with rio.open(
        path_out, 'w',
        driver='GTiff',
        height=im_height,
        width=im_width,
        count=im_bands,
        dtype=im_data.dtype,
        crs = im_crs,
        transform=im_transform,
) as dst:
        dst.write(im_data)
