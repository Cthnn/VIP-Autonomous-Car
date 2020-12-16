This repo is an unfinished repo on my Computer Vision, and arduino tests for an autonomous car. Some work is not made public in this repository because it is still being worked on. 

# What is this repository?
This is a repository for work that I'm doing for a program at Stony Brook University. This program is called VIP(Vertically Integrated Projects). In this program, students work on projects in which they attempt to push the boundaries of their knowledge. One of my passions is computer vision and being able to pursue a project that lies within computer vision has been extremely fun and challenging.I am working under Professor Anurag Purwar, a Mechanical Engineering professor at Stony Brook. This project focuses on Computer Vision theory and Machine Learning. Some of the ideas implemented are lane segmentation, data augmentation, perspective warping, and neural networks.(Neural network is not made public in this repository)
# Arduino
* arduino_receive_test.ino
  * Testing communication between the Jetson Nano and the arduino kit
 # Python
 * Mirror.py
   * A data augmentation script. A simple module that is used to mirror some of our training data
 * arduino_test.py
   * A test script to interact through serial connectors with the arduino
 * lane_detect.py
   * A lane detection script.Using OpenCV's canny edge detector to find lines in the lane
 * segment.py
   * A module to segment the image to focus on a specific section of the road. Since the data has so much irrelevant data, we determined which segment provides the most important info.
 * network.py
   * A prototype of the neural network base off of NVIDIA's End to End machine learning report. Some changes have been made to fit our goal a bit better
 * steering_edited.py
   * A prototype of the road angle calculation script. This uses our new lane segmentation method of converting the colorscale of the image to hsv, then to grayscale inorder to focus make the grayscale image have mainly the image of the road and a mostly black background. We then converted the image to a birds eye view image by shrinking the width of the bottom of the image and applied our min/max angle algorithm. Written with teammates Gautham Botlaguduru and Ziyun Ye.
 * angle_extraction.py 
    * This script is a more formalized version of angle calculation. organized with helper methods for lane segmentation and variables to define the region of interest trapezoid. This script is used to calculate the angle of the road to annotate the data. This script builds on the knowledge learned from steering_edited.py. This algorithim also attempts to improve at finding corners of contours by using the image corners to find the contour corners using a euclidean distance function. This produces more accurate corner points for our drawn contours which in return gives a better calculation of the road angle. Written by Ethan Cheung

  
