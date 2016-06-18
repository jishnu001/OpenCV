#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <iostream>
#include <stdio.h>
#include <string>
using namespace std;
using namespace cv;

/** Function Headers */


/** Global variables */
string window_name = "Capture - Blob detection";

/** @function main */
int main( int argc, const char** argv )
{

  CvCapture* capture;
  Mat im;

  //-- 2. Read the video stream
  capture = cvCaptureFromCAM( -1 );

  if( capture )
  {
    while( true )
    {
      im = cvQueryFrame( capture );


      SimpleBlobDetector detector;

      // Detect blobs.
      std::vector<KeyPoint> keypoints;
      detector.detect( im, keypoints);

      // Draw detected blobs as red circles.
      // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
      Mat im_with_keypoints;
      drawKeypoints( im, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

      // Show blobs
      imshow("keypoints", im_with_keypoints );
      waitKey(10);
    }
  }
}
