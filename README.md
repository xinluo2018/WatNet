# WatNet
A deep ConvNet for surface water mapping based on Sentinel-2 image

## **How to use the trained WatNet?**

### Step 1
- Enter the following commands for 1)download the code files, 2) go to your workspace directory:   

  ~~~console
  git clone https://github.com/xinluo2018/WatNet.git
  ~~~

### Step 2
- Download Sentinel-2 images, and select four visible-near infrared 10-m bands and two 20-m shortwave infrared bands, which corresponding to the band number of 2, 3, 4, 8, 11, and 12 of sentinel-2 image.

### Step 3
- Add the prepared sentinel-2 image (6 bands) to the **_data/test-data(demo)_** directory, and modify the data path in the **_notebooks/infer_demo.ipynb_** file. then surface water mapping can be performed by running the code file: **_notebooks/infer_demo.ipynb_** (this code will run a sample without any modification to be made.)


## **How to train the WatNet?**

- With the Dataset (~will be released shortly), the user can train the WatNet through running the code file **_train/trainer.ipynb_**.  

