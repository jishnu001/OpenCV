#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <ctime>
#include <string>
#include <chrono>
using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame, int &no_faces );

/** Global variables */
String face_cascade_name = "data/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

/** @function main */
int main( int argc, const char** argv )
{
  
  CvCapture* capture;
  Mat frame;
  int no_faces=0;
  //-- 1. Load the cascades
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading file\n"); return -1; };
  if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading file\n"); return -1; };

  //-- 2. Read the video stream
  capture = cvCaptureFromCAM( -1 );

  if( capture )
  {
    while( true )
    {
      frame = cvQueryFrame( capture );

      //-- 3. Apply the classifier to the frame
      if( !frame.empty() )
      { detectAndDisplay( frame,no_faces ); }
      else
      { printf(" --(!) No captured frame -- Break!"); break; }

      int c = waitKey(10);
      if( (char)c == 'c' ) { break; }
    }
  }
  return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, int &no_faces )
{
  std::vector<Rect> faces;
  Mat frame_gray;

  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
  vector<Mat> image_roi;
  int count = faces.size();
  for( size_t i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    //ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
    rectangle(frame,Point(faces[i].x,faces[i].y),Point(faces[i].x+faces[i].width,faces[i].y+faces[i].height),  Scalar( 0, 55, 255 ), +2, 4 );;
    Mat faceROI = frame_gray( faces[i] );
    std::vector<Rect> eyes;

    //-- In each face, detect eyes
    eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    for( size_t j = 0; j < eyes.size(); j++ )
    {
      Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
      int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
      circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
    }
    image_roi.push_back(frame(cv::Rect(Point(faces[i].x,faces[i].y),Point(faces[i].x+faces[i].width,faces[i].y+faces[i].height))));
  }
  std::string s = "Number of faces: ";
  s.append(std::to_string(faces.size()));
  std::string filename = "face";
  //-- Show what you got
  putText(frame, s, Point(50,100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,200,200), 4);
  imshow( window_name, frame );
  auto now = std::chrono::system_clock::now();
  auto now_c = std::chrono::system_clock::to_time_t(now);
  stringstream ext;
  ext << std::put_time(std::localtime(&now_c), "_%Y_%M_%d_%H_%m_%S");
  filename.append(ext.str());
  filename.append(".png");
  std::vector<Mat>::iterator it;

  if(count>no_faces)
  {

    for(it=image_roi.begin();it<image_roi.end();it++){

      imwrite( filename, *it);

    }
      no_faces=count;
  }

}
