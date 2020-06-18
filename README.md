# Image Segmentation with Gaussian Mixture Model

## Authors
Prasheel Renkuntla
Shubham Sonawane
Raj Prakash Shinde
 
## Description
This project deals with the detection of underwater buoys of different colors using Gaussian mixture model. The Expectation Maximization algorithm is applied to learn the color distribution of the three buoys of colors Green, Orange and Yellow. The above procedure is first applied on 1D gaussian(single channel) then on 3D Gaussian distribution of the image.

## Dependencies
* Ubuntu 16
* Python 3.7
* OpenCV 4.2
* Numpy
* matplotlib
* sys
* math


## Run
To run the code for detecting all buoys using 3D gaussian
```
python3.7 model_all_colors_3D.py
```
Enter the choice of code (histogram, training or detection of buoys) when prompted

To run the code for 1D detection of yellow buoy
```
python3.7 yellow_detection.py
```
To run the code for 1D detection of green buoy
```
python3.7 green_detection.py
```
To run the code for 1D detection of orange buoy
```
python3.7 orange_detection.py
```

## Demo
First, we create a dataset from each of the frames of the video and distinguish each colored buoy. A sample orange buoy is given below-
<p align="center">
<h5>Cropped orange buoy</h5>
<img src="/Output/output.PNG" width="70%">
</p>

Based on the average histogram for each colored buoy we decide the number of cluster that would be needed to identify the color distribution. The gaussian distribution for orange colored buoy is given below- 
<p align="center">
<h5>Gaussian distribution of Orange Buoy</h5>
<img src="/Output/Orange-Gaussian.PNG" width="70%">
</p>

Now, after color distribution is known, we use these parameters to detect the buoy individually first, then all three combined on the input video. The top left corner image is the mask obtained from weights learned, the bottom left corner is the edges that we determine from the mask. The next output is the top right image of the dilated version of edges. From this we find the mean and make a circle around it to detect the buoy as below-
<p align="center">
<h5>Pipeline for buoy detection at 140th frame</h5>
<img src="/Output/yellow_140.png" width="70%">
</p>

The final image below shows the progressive detection-
<p align="center">
<h5>Buoy detection for yellow, orange and green buoys</h5>
<img src="/Output/allThree.png" width="70%">
</p>

The Output folder has video results from 3D GMM only.

## Reference
* https://cmsc426.github.io/colorseg/#colorclassification
* https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95
* https://towardsdatascience.com/gaussian-mixture-modelling-gmm-833c88587c7f

