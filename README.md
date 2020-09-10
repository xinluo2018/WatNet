# Earth Surface Water Mapping with Satellite Image
**_We recommend the user run these code on google colab platform, thus the tedious environment configuration is not required for your local computer._** 
## Step 1
- Go to your google drive, create a new colab file, and then enter the commands for 1)Mount on google drive, 2) go to your root directory:    
*from google.colab import drive  
drive.mount('/content/drive/')  
import os    
os.chdir("/content/drive/My Drive")*  

## Step 2
-  Enter the following commands for 1)download the code files, 2)go to your work directory:   
*!git clone https://github.com/xinluo2018/Earth-surface-water-mapping.git  
os.chdir("/content/drive/My Drive/Earth-surface-water-mapping")*


## Step 3
- Surface water mapping through running the code file **_infer_demo.ipynb_**, the user can replace your own sentinel-2 image in the test_image_demo directory. Bands order is blue-green-red-nir-swir1-swir2, and the 20-m bands should be downsampled to 10-m resolution.   
- With the Dataset, the user also can train the WatNet through running the code file **_trainer.ipynb_**.  
