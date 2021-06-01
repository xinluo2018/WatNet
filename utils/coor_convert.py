## ref: https://blog.csdn.net/Prince999999/article/details/105843511

import numpy as np
from osgeo import osr

def getSRSPair(dataset):
    '''
    :param dataset: GDAL data (read by gdal.Open())
    :return: projection and georeference information
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs


def geo2imagexy(lon,lat,img_gdal):
    '''
    description: from georeferenced location (i.e., lon, lat) to image location(col,row).
    :param img_gdal: GDAL data (read by gdal.Open()))
    :param lon: project or georeferenced x, i.e.,lon
    :param lat: project or georeferenced y, i.e., lat
    :return: image col and row corresponding to the georeferenced location.
    '''
    trans = img_gdal.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([lon - trans[0], lat - trans[3]])
    col_fps, row_fps = np.linalg.solve(a, b)
    col_fps, row_fps = np.floor(col_fps).astype('int'), np.floor(row_fps).astype('int')
    return col_fps, row_fps

def imagexy2geo(dataset, row, col):
    '''
    :dataset: GDAL data (read by gdal.Open())
    :row and col are corresponding to input image (dataset)
    :return:  geographical coordinates
    '''
    trans = dataset.GetGeoTransform()
    px = trans[0] + col * trans[1] + row * trans[2]
    py = trans[3] + col * trans[4] + row * trans[5]
    return px, py


