# **Traffic Sign Recognition** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/count_of_each_sign.png "Visualization"
[image2]: ./examples/sign_before_process.png "Before"
[image3]: ./examples/sign_after_process.png "After"
[image4]: ./examples/new1_small.png "Traffic Sign 1-5"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized the image data because I want the training data to have ~0 mean and small stddev. The normalized images have pixels value ranging from [-1, 1]. But the image still have three channels of colors R, G, and B

As the second step, I decided to convert the images to grayscale because the model of a depth = 8 first convolution layer and depth = 20 second convolution layer doesn't perform well with 3-channel color images. And the color info doesn't really help to recognize traffic signs.

Here is an example of a traffic sign image before and after normalization and grayscaling. 

![alt text][image2] ![alt text][image3]

The difference between the original data set and the augmented data set is the following ...
* The original data have three color channel but the augmented data have only one color channel
* The original data have pixel value ranging from [0, 255], and the augmented data
have pixel value ranging from [-1, 1]
* The original data have pixel value with mean of ~128, and the augmented data
have pixel value with mean of ~0

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GRAY image   							| 
| Convolution 3x3     	| 5x5 stride, valid padding, outputs 28x28x8 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x8	 				|
| Convolution 3x3	    | 5x5 stride, valid padding, outputs 10x10x20	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x20	 				|
| Fully connected		| outputs 120  									|
| RELU					| 			  									|
| Dropout				| keep_probability = 0.5						|
| Fully connected		| outputs 84  									|
| RELU					| 			  									|
| Dropout				| keep_probability = 0.5						|
| Fully connected		| outputs 43  									|
| Logits				| outputs 43   									|
| Softmax				|  		      									|


#### 3. Training the Model

To train the model, I used Stochastic Gradient Descent optimized by AdamOptimizer at a learning rate of 0.001. Each batch was a randomized sample of 128 training samples. The training takes 12 Epoches

#### 4. Tuning the Model

The approach to classify the traffic symbols was to implement a standard LeNet-5 CNN and iteratively tune it to improve performance for this specific dataset. The LeNet-5 model comprises of a stack of two convolution layers and three fully connected layers with two RELU activations interleaved betweeen them. The convolutions layers outputs are also fed through Max-Pooling layers after RELU. One of the changes that improved performance for this dataset is the inclusion of dropout layers connected to fully-connected layers. This was added when I noticed the model was overfitting to the training data set. The training accuracy is very high but the validation accuracy is low, which implies overfitting

* Convolution layer depth, learning rate, batch size, epochs and the keep probablity for the dropout layers are the most important hyperparameters that I had to tune.
* My initial learning rate was 0.01, but the validation accuracy goes to around 0.80 at the first several epoches and then stay there

My final model results were:
* training set accuracy of 0.988
* validation set accuracy of 0.956 
* test set accuracy of 0.936 

### Test Model on New Images

#### 1. Here are five German traffic signs that I found on the web:

![alt text][image4] 

The first image might be difficult to classify because ...

#### 2. Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children crossing		| Children crossing								| 
| No passing   			| No passing 									|
| Yield					| Yield											|
| 30 km/h	      		| 30 km/h						 				|
| stop					| stop    										|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 1.0

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 26th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a children crossing sign (probability of 1.0), and the image does contain a children crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9999960     			| Children crossing  							| 
| .0000039  			| Bicycle crossing 								|
| .0000000				| Beware of ice/snow							|
| .0000000	   			| Road narrows on the right		 				|
| .0000000			    | Slippery Road      							|

For the second image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000000     			| No passing 									| 
| .0000000  			| Vehicles over 3.5 metric tons prohibited		|
| .0000000				| No passing for vehicles over 3.5 metric tons	|
| .0000000	   			| Slippery Road 				 				|
| .0000000			    | End of no passing by vehicles over 3.5 metric |

For the third image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000000    			| Yield  										| 
| .0000000  			| Ahead only									|
| .0000000				| Beware of ice/snow							|
| .0000000	   			| Road narrows on the right		 				|
| .0000000			    | Slippery Road      							|

For the forth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9998394     			| 30 km/h  										| 
| .0001604  			| 20 km/h 										|
| .0000000				| 50 km/h										|
| .0000000	   			| 60 km/h						 				|
| .0000000			    | 70 km/h 		    							|

For the fifth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9999960     			| stop 											| 
| .0000039  			| 30 km/h 										|
| .0000000				| No Vehicles									|
| .0000000	   			| Yield		 									|
| .0000000			    | 70 km/h 		    							|

### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

From the generated feature maps for the first convolutional layer, we can notice that the model mainly rely on oval and curves (for circular signs) and edges (for triangular signs) of the traffic sign. 

From the generated feature maps for the second convolutional layer, we can notice that the model mainly rely on the lightness pattern of the traffic sign.


