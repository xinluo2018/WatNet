import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def acc_matrix(cla_map, sam_pixel, id_label):
    '''
    Arguments: 
            cla_map: classification result of the full image
            sam_pixel: array(num_samples,3), col 1,2,3 are the row,col and label.
            id_label: 0,1,2...
    Return: 
            the overall accuracy and confusion matrix
    '''
    sam_result = []
    num_cla = sam_pixel[:,2].max()+1
    labels = list(range(num_cla))
    for i in range(sam_pixel.shape[0]):
        sam_result.append(cla_map[sam_pixel[i,0], sam_pixel[i,1]])
    sam_result = np.array(sam_result)
    acc_oa = np.around(accuracy_score(sam_pixel[:,2], sam_result), 4)
    confus_mat = confusion_matrix(sam_pixel[:,2], sam_result, labels=labels)
    acc_prod=np.around(confus_mat[id_label,id_label]/confus_mat[id_label,:].sum(), 4)
    acc_user=np.around(confus_mat[id_label,id_label]/confus_mat[:,id_label].sum(), 4)

    return acc_oa, acc_prod, acc_user, confus_mat

