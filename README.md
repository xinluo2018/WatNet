# -- WatNet
A deep ConvNet for surface water mapping based on Sentinel-2 image

![watnet](figures/watnet_structure.png)



# -- WatSet
- Surface water dataset for Deep learning could be downloaded from:  
[Google Drive](https://drive.google.com/drive/folders/1f8HPAe2wBUga-ImiYnFxaGlHGzvMDKF4?usp=sharing) and [Baidu Drive](https://pan.baidu.com/s/1V-k3me1gT0ph4kRmNrDnIw) (Fetch Code: uic2)  


<div align="left">
<img src=figures/dataset.png height="310px" alt=distribution >
</div>

|Labeling example 1:|Labeling example 2:|
| :-- | :-- |
| ![example_1](figures/label_sam_1.png)| ![example_2](figures/label_sam_2.png)|

# -- performance
- Examples for surface water mapping  
  **Urban region**  
  <!-- <div align="left">
  <img src=figures/urban/urban-scene.png height="100px" alt= urban_scene>
  <img src=figures/urban/urban-awei.png height="100px" alt= urban_awei>
  <img src=figures/urban/urban-mndwi.png height="100px" alt= urban_mndwi >
  <img src=figures/urban/urban-watnet.png height="100px" alt= urban_watnet>
  </div> -->
  |||||  
  |--|--|--|--|  
  |![urban_scene](figures/urban/urban-scene.png)|![urban-awei](figures/urban/urban-awei.png)|![urban-mndwi](figures/urban/urban-mndwi.png)|![urban-watnet](figures/urban/urban-watnet.png)|


  **Cloudy region**  
  <!-- <div align="left">
  <img src=figures/cloudy/cloudy-scene.png height="100px" alt= cloudy_scene>
  <img src=figures/cloudy/cloudy-awei.png height="100px" alt= cloudy_awei>
  <img src=figures/cloudy/cloudy-obia.png height="100px" alt= cloudy_obia >
  <img src=figures/cloudy/cloudy-watnet.png height="100px" alt= cloudy_watnet>
  </div> -->
  |||||  
  |--|--|--|--|  
  |![cloudy_scene](figures/cloudy/cloudy-scene.png)|![cloudy-awei](figures/cloudy/cloudy-awei.png)|![cloudy-mndwi](figures/cloudy/cloudy-obia.png)|![cloudy-watnet](figures/cloudy/cloudy-watnet.png)|



  **Mountainous region**  
  <!-- <div align="left">
  <img src=figures/mountain/mountain-scene.png height="100px" alt= mountain_scene>
  <img src=figures/mountain/mountain-awei.png height="100px" alt= mountain_awei>
  <img src=figures/mountain/mountain-obia.png height="100px" alt= mountain_obia >
  <img src=figures/mountain/mountain-watnet.png height="100px" alt= mountain_watnet> -->
  <!-- </div> -->
  |||||  
  |--|--|--|--|  
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

```python
   from watnet_infer import watnet_infer   
   water_map = watnet_infer(rsimg)  # full example in notebooks/infer_demo.ipynb.
```
- --- command line API:
  ~~~console
  python watnet_infer.py data/test-demo/*.tif -o data/test-demo/result
  ~~~




## **-- How to train the WatNet?**

- With the Dataset (~will be released shortly), the user can train the WatNet through running the code file **_train/trainer.ipynb_**.  

## -- Acknowledgement  
We thanks the authors for providing some of the code in this repo:  
[deeplabv3_plus](https://github.com/luyanger1799/amazing-semantic-segmentation)  
[deepwatmapv2](https://github.com/isikdogan/deepwatermap)  

