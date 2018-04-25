[aachen street image]:https://github.com/teeekay/CarND-Semantic-Segmentation/blob/master/examples/aachen_000053_000019_leftImg8bit.png?raw=true "Aachen Cityscapes image 53"
[aachen labelled image]:https://github.com/teeekay/CarND-Semantic-Segmentation/blob/master/examples/aachen_000053_000019_gtFine_color.png?raw=True "Aachen Cityscapes labelled image 53"
[um_000002]:https://github.com/teeekay/CarND-Semantic-Segmentation/blob/master/examples/um_000002.png?raw=true "KITTI Road um_000002"
[um_000011]:https://github.com/teeekay/CarND-Semantic-Segmentation/blob/master/examples/um_000011.png?raw=true "KITTI Road um_000011"
[um_000089]:https://github.com/teeekay/CarND-Semantic-Segmentation/blob/master/examples/um_000089.png?raw=true "KITTI Road um_000089"
[umm_000002]:https://github.com/teeekay/CarND-Semantic-Segmentation/blob/master/examples/umm_000002.png?raw=true "KITTI Road umm_000002"
[umm_000007]:https://github.com/teeekay/CarND-Semantic-Segmentation/blob/master/examples/umm_000007.png?raw=true "KITTI Road umm_000007"
[umm_000012]:https://github.com/teeekay/CarND-Semantic-Segmentation/blob/master/examples/umm_000012.png?raw=true "KITTI Road umm_000012"
[uu_000001]:https://github.com/teeekay/CarND-Semantic-Segmentation/blob/master/examples/uu_000001.png?raw=true "KITTI Road uu_000001"
[uu_000011]:https://github.com/teeekay/CarND-Semantic-Segmentation/blob/master/examples/uu_000011.png?raw=true "KITTI Road uu_000011"

# Semantic Segmentation
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
### Operation
main.py supports 3 modes of operation: 1) run training, 2) process images based on an inference model, 3) process a video based on an inference model

To run training for 15 epochs with a batch size of 1 and a learning rate of 0.00015 and save the results in model1.meta
 
```sh
python main.py -md=0 -ep=30 -bs=1 -lr=0.000015 -mod='model1'
```
To generate the inference samples from the Kitti Road dataset using model1.meta
```sh
python main.py -md=1 -mod='model1'
```
to run inference on a video using model1.meta
```sh
python main.py -md=2 -mod='model1'
```

### Development

In this application, labelled training images taken from the [Cityscapes](https://www.cityscapes-dataset.com/) dataset were used to train a fully convolutional model using the [FCN-8](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) architecture.  FCN-8 uses the VGG16 encoder which has been trained on Imagenet for classification.  A fully convolutional decoder is added which combines pool layers 3 and 4 and fully connected layer7 from the encoder to enable pixel level classification of images.

In each epoch 500 images are randomly selected from the dataset of 2,876 and a different set of 500 images are used to calculate Intersection over Union.  `adjust_cityscapes.py` was used to downscale and crop the source images to the model size of 576 pixels wide by 160 high.  The labelled images were also processed to match those provided with the KITTI dataset in this assignment, where all pixels that did not match the purple color (RGB=128,64,128) assigned to roads by Cityscapes were set to red (RGB=255,0,0).


| Example Image from Cityscapes Aachen Dataset
|-|
| ![alt-text][aachen street image] |
| Aachen Cityscapes image 19 |
| ![alt-text][aachen labelled image] |
| Road pixels labelled in purple - everything else Red |


After training, the inference model was used on the KITTI images which were not used during training.  A gradational mask of green pixels was overlaid onto the original images based on the softmax probability that the pixel belonged to road (decreasing transparency as probability increases above 0.25, 0.5, and 0.75).  In most cases the gradation was quite abrupt, with sharp definition of areas the model assigned as road. 


|Images from KITTI dataset with road predictions labelled|
|-|
|![][um_000002]|
|![][um_000011]|
|![][um_000089]|
|![][umm_000002]|
|![][umm_000007]|
|![][umm_000012]|
|![][uu_000001]|
|![][uu_000011]|


A video was also processed with acceptable results.

[<img src="https://github.com/teeekay/CarND-Semantic-Segmentation/blob/master/examples/movie.jpg?raw=true" width = 300>](https://youtu.be/NhGzExWjcDM)


## Training

The Adam optimizer was used in combination with a learning rate of 0.000015.  The model was run for 15 epochs of 500 steps each, with each step using a batch size of 1.  I was limited to using this batch size as my GPU would not support larger batch sizes.

Intersection over Union (IOU) was computed after each step along with cross entropy loss and regularization losses.

A plot of IOU increasing and loss decreasing as the model was trained is shown below.


<img src="https://github.com/teeekay/CarND-Semantic-Segmentation/blob/master/examples/iou_and_xentloss.png?raw=true" width=700> 

## Model Structure as displayed in Tensorboard

<img src="https://github.com/teeekay/CarND-Semantic-Segmentation/blob/master/examples/tf_graph.png?raw=true" width=700 alt="tensorflow model in tensorboard">  




## From Udacity
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
