# -- WatNet
- A deep ConvNet for surface water mapping based on Sentinel-2 image

  ![watnet](figures/watnet_structure.png)



# -- DataSet
- Surface water dataset for Deep learning could be downloaded from:  
[**Google Drive**](https://drive.google.com/drive/folders/1f8HPAe2wBUga-ImiYnFxaGlHGzvMDKF4?usp=sharing) and [**Baidu Drive**](https://pan.baidu.com/s/1V-k3me1gT0ph4kRmNrDnIw) (Fetch Code: uic2)  

  ![dataset](figures/dataset.png)

  |Labeling example 1:|Labeling example 2:|
  | :-- | :-- |
  | ![example_1](figures/label_sam_1.png)| ![example_2](figures/label_sam_2.png)|

# -- Performance
- Examples for surface water mapping  

  **Urban region**  
  |Urban scene|AWEI|MNDWI|WatNet|  
  |:--|:--|:--|:--|  
  |![urban_scene](figures/urban/urban-scene.png)|![urban-awei](figures/urban/urban-awei.png)|![urban-mndwi](figures/urban/urban-mndwi.png)|![urban-watnet](figures/urban/urban-watnet.png)|

  **Cloudy region**  
  |Cloudy scene|AWEI|MNDWI|WatNet|  
  |:--|:--|:--|:--|  
  |![cloudy_scene](figures/cloudy/cloudy-scene.png)|![cloudy-awei](figures/cloudy/cloudy-awei.png)|![cloudy-mndwi](figures/cloudy/cloudy-obia.png)|![cloudy-watnet](figures/cloudy/cloudy-watnet.png)|

  **Mountainous region**  
  |Mountain scene|AWEI|MNDWI|WatNet|  
  |:--|:--|:--|:--|  
  |![mountain_scene](figures/mountain/mountain-scene.png)|![mountain-awei](figures/mountain/mountain-awei.png)|![mountain-mndwi](figures/mountain/mountain-obia.png)|![mountain-watnet](figures/mountain/mountain-watnet.png)|


## **-- How to use the trained WatNet?**

### -- Step 1
- Enter the following commands for downloading the code files, and then configure the python and deep learning environment. The deep learning software used in this repo is [Tensorflow 2](https://www.tensorflow.org/).

  ~~~console
  git clone https://github.com/xinluo2018/WatNet.git
  ~~~

### -- Step 2
- Download Sentinel-2 images, and select four visible-near infrared 10-m bands and two 20-m shortwave infrared bands, which corresponding to the band number of 2, 3, 4, 8, 11, and 12 of sentinel-2 image.

### -- Step 3
- Add the prepared sentinel-2 image (6 bands) to the **_data/test-demo_** directory, modify the data name in the **_notebooks/infer_demo.ipynb_** file, then running the code file: **_notebooks/infer_demo.ipynb_** and surface water map can be generated. 
- Users also can specify the program for surface water mapping by using the watnet_infer.py, specifically,  
- --- funtional API:

  ~~~python
  from watnet_infer import watnet_infer   
  water_map = watnet_infer(rsimg)  # full example in notebooks/infer_demo.ipynb.
  ~~~
- --- command line API:
  ~~~console
  python watnet_infer.py data/test-demo/*.tif -o data/test-demo/result
  ~~~

## **-- How to train the WatNet?**

- With the Dataset (~will be released shortly), the user can train the WatNet through running the code file **_train/trainer.ipynb_**.  

## -- Citation (~~ on publishment)



## -- Acknowledgement  
We thanks the authors for providing some of the code in this repo:  
[deeplabv3_plus](https://github.com/luyanger1799/amazing-semantic-segmentation) and [deepwatmapv2](https://github.com/isikdogan/deepwatermap)  

