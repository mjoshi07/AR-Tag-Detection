# AR-Tag-Detection
The objective is to detect a custom AR tag that can be used to obtain a point of reference in the real world for augemented reality applications

## Overview
* The project is divided into 3 parts
* In First part, the AR Tag is detected and it tag orientation and ID is decoded
* In the second part, a reference image is placed on the Tag such that it superimposes the tag in the whole video
* In the third part, a virtual cube is placed on top of the Tag in the whole video
* OpenCV methods such as find-contours, hough-transform, find-homography and warp-perspective have not been used, these methods have been implemented from scratch 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Results
* AR Tag Corners Detected and ID Decoded: 0111 or 7
<p align="center">
<img src="https://github.com/mjoshi07/AR-Tag-Detection/blob/main/Data/results/AR_tag.png" width=640/>
</p>

* Testudo Image superimposed on the detected AR Tag
<p align="center">
<img src="https://github.com/mjoshi07/AR-Tag-Detection/blob/main/Data/results/img_on_AR_tag.gif"/>
</p>

* Virtual Cube placed on the detected AR Tag
<p align="center">
  <img src="https://github.com/mjoshi07/AR-Tag-Detection/blob/main/Data/results/cube2.png" height = 150/>
  <img src="https://github.com/mjoshi07/AR-Tag-Detection/blob/main/Data/results/cube1.png"  height = 150/>
  <img src="https://github.com/mjoshi07/AR-Tag-Detection/blob/main/Data/results/cube3.png"  height = 150/>
</p>

## Build dependencies
* Numpy
* Imutils
* Opencv
* Matplotlib
* Argparse

## Run Instructions
* Clone the repo
```
git clone https://github.com/mjoshi07/AR-Tag-Detection.git
```
* Cd to the Code Directory
```
cd AR-Tag-Detection/Code
```
* Run the following command to see the options
```
python main.py -h
```
* You will see the following options
```
optional arguments:
  -h, --help            show this help message and exit
  --problem PROBLEM     problem number you want to solve
  --dataPath DATAPATH   path to data folder, Default: ..//data//
  --videoFile VIDEOFILE video file name, Default: 1tagvideo.mp4
  --imgFile IMGFILE     image file name, Default: testudo.png
```
* To solve the problem 1 run the following command
```
python main.py --problem 1 --dataPath "path//to//data//directory//" --videoFile "Video_filename_with_extension" 
```
* To Solve the problem 2(a) run the following command
```
python main.py --problem 2 --dataPath "path//to//data//directory//" --videoFile "Video_filename_with_extension" --imgFile "Img_filename_with_extension"
```
* To Solve the problem 2(b) run the following command
```
python main.py --problem 3 --dataPath "path//to//data//directory//" --videoFile "Video_filename_with_extension"
```



