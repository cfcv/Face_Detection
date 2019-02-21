# Face_Detection
## Requirements
To run this code you must have installed opencv 3.0 version or above and also the opencv_contrib. To execute the code if you are in linux you can just run the command bellow:
* `g++ -std=c++11 faceDetection.cpp ``pkg-config --libs --cflags opencv`` -o faceDetection`


Obs: with only one ` in each side;

Otherwise, you have to link the opencv .dll or .so to the project(using visual studio, eclipse, etc.).

## Description
This projet test the deep learning based face detector incorporated by opencv since the 3.0 version, using the dnn module. This model overcomes all the drawbacks of the Haar cascade based face detector used in the previous versions, it detect faces in many diferent poses(up, down, left, right...) with a pretty good accuracy, in various scales and even with considerable oclusion. Futhermore, it runs also in real-time CPU.

## Results
**Some results:**

![alt text](https://github.com/cfcv/Face_Detection/blob/master/result_photos/result_up.png) ![alt text](https://github.com/cfcv/Face_Detection/blob/master/result_photos/result_down.png)
![alt text](https://github.com/cfcv/Face_Detection/blob/master/result_photos/result_right.png) ![alt text](https://github.com/cfcv/Face_Detection/blob/master/result_photos/result_left.png)

**With occlusion:**

![alt text](https://github.com/cfcv/Face_Detection/blob/master/result_photos/oclusion.png) ![alt text](https://github.com/cfcv/Face_Detection/blob/master/result_photos/oclusion_2.png)
