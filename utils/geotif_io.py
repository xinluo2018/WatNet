## author: luo xin, date: 2021.6.18

import numpy as np
from osgeo import gdal
from osgeo import osr

### tiff image reading
def readTiff(path_in):
    '''
    return: 
        img: numpy array, exent: tuple, (x_min, x_max, y_min, y_max) 
        proj info, and dimentions: (row, col, band)
    '''
    RS_Data=gdal.Open(path_in)
    im_col = RS_Data.RasterXSize  # 
    im_row = RS_Data.RasterYSize  # 
    im_bands =RS_Data.RasterCount  # 
    im_geotrans = RS_Data.GetGeoTransform()  # 
    im_proj = RS_Data.GetProjection()  # 
    img_array = RS_Data.ReadAsArray(0, 0, im_col, im_row)  # 
    left = im_geotrans[0]
    up = im_geotrans[3]
    right = left + im_geotrans[1] * im_col + im_geotrans[2] * im_row
    bottom = up + im_geotrans[5] * im_row + im_geotrans[4] * im_col
    extent = (left, right, bottom, up)
    espg_code = osr.SpatialReference(wkt=im_proj).GetAttrValue('AUTHORITY',1)

    img_info = {'geoextent': extent, 'geotrans':im_geotrans, \
                'geosrs': espg_code, 'row': im_row, 'col': im_col,\
                    'bands': im_bands}

    if im_bands > 1:
        img_array = np.transpose(img_array, (1, 2, 0)).astype(np.float)  # 
        return img_array, img_info 
    else:
        return img_array, img_info

###  .tiff image write
def writeTiff(im_data, im_geotrans, im_geosrs, path_out):
    '''
    input:
        im_data: tow dimentions (order: row, col),or three dimentions (order: row, col, band)
        im_geosrs: espg code correspond to image spatial reference system.
    '''
    im_data = np.squeeze(im_data)
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
        dataset.SetGeoTransform(im_geotrans)       # 
        dataset.SetProjection("EPSG:" + str(im_geosrs))      # 
    if im_bands > 1:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])
        del dataset
    else:
        dataset.GetRasterBand(1).WriteArray(im_data)
        del dataset

# import rasterio as rio
# import numpy as np

# def readTiff(path_in):
#     ''' 
#         author: xin luo, date: 2021.5.6
#         input: path of the .tif image
#         output: rasterio DatasetReader object, and np.array data (image)
#     '''
#     img_src = rio.open(path_in)
#     if img_src.count > 1:
#         bands_ind = [i for i in range(1, img_src.count+1)]
#         img_array = img_src.read(bands_ind)
#         img_array = img_array.transpose(1, 2, 0)
#     else: 
#         img_array = img_src.read(1)
#     return img_src, img_array


# def writeTiff(im_data, im_transform, im_crs, path_out):
#     '''im_data: 3d (row, col, band) np.array.'''
#     if len(im_data.shape) > 2:
#         im_data = im_data.transpose(2,0,1)
#         im_bands, im_height, im_width = im_data.shape
#     else:
#         im_bands, (im_height, im_width) = 1, im_data.shape
#         im_data = np.expand_dims(im_data, axis=0)
#     with rio.open(
#         path_out, 'w',
#         driver='GTiff',
#         height=im_height,
#         width=im_width,
#         count=im_bands,
#         dtype=im_data.dtype,
#         crs = im_crs,
#         transform=im_transform,
# ) as dst:
#         dst.write(im_data)