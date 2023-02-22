# Image Segmentation using a U-Net

* forest_segmentation: binary segmentation
    * data: https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation
    * model: unet_segmentation.py
    * input is expected to be in folders 'masks' and 'images' as .jpg images

* cityscapes-images-paris: multiple segmentation
    * data: https://www.kaggle.com/datasets/dansbecker/cityscapes-image-pairs
    * preprocessing of the multiclass masks has been adapted from: https://www.kaggle.com/code/yauhenikavaliou/camseq-semantic-segmentation
    * model: unet_segmentation_multi.py
    * input is expected to be in folders 'masks' and images

* Possible improvements:
    * add class weights 
