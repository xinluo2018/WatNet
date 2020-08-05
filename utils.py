#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# mount on google drive
from google.colab import drive
drive.mount('/content/drive/')
# to the user's work directory
import os
os.chdir("/content/drive/My Drive/Colab/WaterMapping/Github_upload")
# !ls
# !nvidia-smi


# In[ ]:


try:
    get_ipython().magic(u'tensorflow_version 2.x')
except Exception:
    pass
import tensorflow as tf
from osgeo import gdal
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[ ]:


### .tiff image reading
def readTiff(path_in):
    '''
    return: numpy array, dimentions order: (row, col, band)
    '''
    RS_Data=gdal.Open(path_in)
    im_col = RS_Data.RasterXSize  # 
    im_row = RS_Data.RasterYSize  # 
    im_bands =RS_Data.RasterCount  # 
    im_geotrans = RS_Data.GetGeoTransform()  # 
    im_proj = RS_Data.GetProjection()  # 
    RS_Data = RS_Data.ReadAsArray(0, 0, im_col, im_row)  # 
    if im_bands > 1:
        RS_Data = np.transpose(RS_Data, (1, 2, 0)).astype(np.float)  # 
        return RS_Data, im_geotrans, im_proj, im_row, im_col, im_bands
    else:
        return RS_Data,im_geotrans,im_proj,im_row,im_col,im_bands

###  .tiff image write
def writeTiff(im_data, im_geotrans, im_proj, path_out):
    '''
    im_data: tow dimentions (order: row, col),or three dimentions (order: row, col, band)
    '''
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_data = np.transpose(im_data, (2, 0, 1))
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands,(im_height, im_width) = 1,im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path_out, im_width, im_height, im_bands, datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans)    # 
        dataset.SetProjection(im_proj)      # 
    if im_bands > 1:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])
        del dataset
    else:
        dataset.GetRasterBand(1).WriteArray(im_data)
        del dataset


# In[ ]:


## accuray assessemnt functions
def acc_patch(Patch_Truth, outp_Patch):
    '''
    input: patch truth and patch output
    return: overall accuracy and mean IoU
    '''
    outp_Patch = tf.where(outp_Patch > 0.5, 1, 0)
    m_OA = tf.keras.metrics.BinaryAccuracy()
    m_OA.update_state(Patch_Truth, outp_Patch)
    Acc_OA = m_OA.result().numpy()
    m_MIoU = tf.keras.metrics.MeanIoU(num_classes=2)
    m_MIoU.update_state(Patch_Truth, outp_Patch)
    Acc_MIoU = m_MIoU.result().numpy()
    return Acc_OA, Acc_MIoU

def get_sample(path_sam, label):
    '''
    Arguments: 
        path_sam_cla: excel file (one class) export from envi
        label: an integer for the specific class.
    Return: 
        np.array contains location and label (order: (row, col, labe)) of 
        the smaples    
    '''
    sam = np.loadtxt(path_sam, dtype=np.str, delimiter=",",skiprows=(9))
    sam = sam[:,(1,0)].astype(int)
    sam = np.pad(sam, ((0, 0), (0, 1)), 'constant', constant_values=label)
    return sam

def acc_sample(cla_map,sam):
    '''
    Arguments: 
            cla_map: classification result of the full image
            sam: array(num_samples,3), col 1,2,3 are the row,col and label of the testing samples.
    Return: 
            the overall accuracy and confusion matrix
    '''
    sam_result = []
    for i in range(sam.shape[0]):
        sam_result.append(cla_map[sam[i,0],sam[i,1]])
    sam_result = np.array(sam_result)
    acc = np.around(accuracy_score(sam[:,2],sam_result),4)
    confus_mat = confusion_matrix(sam[:,2],sam_result) # 
    confus_mat_per = confus_mat/np.tile(np.sum(confus_mat, axis = 0),(confus_mat.shape[0],1))  # producer's accuracy 
    return acc, confus_mat_per

