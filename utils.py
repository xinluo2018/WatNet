#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 挂载google drive
from google.colab import drive
drive.mount('/content/drive/')
# 切换工作路径
import os
os.chdir("/content/drive/My Drive/Colab/WaterMapping/Github_upload")
get_ipython().system(u'ls')
# !nvidia-smi


# In[3]:


try:
    # %tensorflow_version only exists in Colab.
    get_ipython().magic(u'tensorflow_version 2.x')
except Exception:
    pass
import tensorflow as tf
from osgeo import gdal
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[4]:


### 读写影像
def readTiff(path_in):
    RS_Data=gdal.Open(path_in)
    im_col = RS_Data.RasterXSize  # 栅格矩阵的列数
    im_row = RS_Data.RasterYSize  # 栅格矩阵的行数
    im_bands =RS_Data.RasterCount  # 波段数
    im_geotrans = RS_Data.GetGeoTransform()  # 获取仿射矩阵信息
    im_proj = RS_Data.GetProjection()  # 获取投影信息
    RS_Data = RS_Data.ReadAsArray(0, 0, im_col, im_row)  # 获取参考影像数据数据
    if im_bands > 1:
        RS_Data = np.transpose(RS_Data, (1, 2, 0)).astype(np.float)  # 调整维度顺序，调整为行，列，波段数
        return RS_Data, im_geotrans, im_proj, im_row, im_col, im_bands
    else:
        return RS_Data,im_geotrans,im_proj,im_row,im_col,im_bands

###  保存含有坐标投影信息的tif函数,输入影像数据存储格式为行,列，波段数，或行，列
def writeTiff(im_data, im_geotrans, im_proj, path_out):
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
        #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path_out, im_width, im_height, im_bands, datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans)    # 写入仿射变换参数
        dataset.SetProjection(im_proj)      # 写入投影
    if im_bands > 1:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])
        del dataset
    else:
        dataset.GetRasterBand(1).WriteArray(im_data)
        del dataset


# In[5]:


def acc_patch(Patch_Truth, outp_Patch):
    outp_Patch = tf.where(outp_Patch > 0.5, 1, 0)
    m_OA = tf.keras.metrics.BinaryAccuracy()
    m_OA.update_state(Patch_Truth, outp_Patch)
    Acc_OA = m_OA.result().numpy()
    m_MIoU = tf.keras.metrics.MeanIoU(num_classes=2)
    m_MIoU.update_state(Patch_Truth, outp_Patch)
    Acc_MIoU = m_MIoU.result().numpy()
    return Acc_OA, Acc_MIoU


# In[6]:


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
    Arguments: sample for testing: array(num_samples,3),
          col 1,2,3 are the row,col and label of the testing samples.
    Return: the overall accuracy and confusion matrix
    '''
    sam_result = []
    for i in range(sam.shape[0]):
        sam_result.append(cla_map[sam[i,0],sam[i,1]])
    sam_result = np.array(sam_result)
    acc = np.around(accuracy_score(sam[:,2],sam_result),3)
    confus_mat = confusion_matrix(sam[:,2],sam_result) # 
    confus_mat_per = confus_mat/np.tile(np.sum(confus_mat, axis = 0),(confus_mat.shape[0],1))  # producer's accuracy 
    return acc, confus_mat_per

