#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

const float confidenceTreshold = 0.6;

int main(){
    //Tensorflow: path to the required files configurations of the neural net
    const string tensorflowConfigFile =  "/home/cfcv/opencv/samples/dnn/face_detector/opencv_face_detector.pbtxt";
    const string tensorflowWeightFile = "/home/cfcv/opencv/samples/dnn/face_detector/opencv_face_detector_uint8.pb";
    //Coffee files
    //const string caffeConfigFile = "/home/cfcv/opencv/samples/dnn/face_detector/deploy.prototxt";
    //const string caffeWeightFile = "/home/cfcv/opencv/samples/dnn/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel";
    
    //Reading the image ang getting the widthe and height
    Mat frame = imread("/home/cfcv/Desktop/down.jpg", IMREAD_COLOR);
    float frameWidth = frame.cols;
    float frameHeight = frame.rows;

    //reading the neural net
    dnn::Net net = dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);
    //Coffee: dnn::Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
    
    //preparing data do pass to the network
    Mat inputBlob = dnn::blobFromImage(frame, 1.0, Size(300,300), Scalar(104.0, 177.0, 123.0), true, false);
    //Coffee: Mat inputBlob = dnn::blobFromImage(frame, 1.0, Size(300,300), Scalar(104.0, 177.0, 123.0), false, false);
    
    //forwarding the input throught the network
    net.setInput(inputBlob, "data");
    Mat detection = net.forward("detection_out");
    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    for(int i = 0; i < detectionMat.rows; ++i){
        float confidence = detectionMat.at<float>(i, 2);
        if(confidence > confidenceTreshold){
            //if we are confidenc that it is a face we draw a rectangle
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
            rectangle(frame, cv::Point(x2, y2), cv::Point(x1, y1), cv::Scalar(0, 255, 0),6);
        }
    }

    namedWindow("Face Detection", CV_WINDOW_NORMAL);
    imshow("Face Detection", frame);
    waitKey();
    return 0;
}
