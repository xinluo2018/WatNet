import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

## Accuray assessemnt functions
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
        np.array contains location and label (order: (row, col, label)) of 
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