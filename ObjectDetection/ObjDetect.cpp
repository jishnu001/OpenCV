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

  //. Read the video stream
  capture = cvCaptureFromCAM( -1 );

  if( capture )
  {
    while( true )
    {
      im = cvQueryFrame( capture );



      SimpleBlobDetector::Params params;
      params.filterByColor = true;
      params.blobColor = 255;
            // Change thresholds
      params.minThreshold = 10;
      params.maxThreshold = 200;

      // Filter by Area.
      params.filterByArea = true;
      params.minArea = 1500;

      // Filter by Circularity
      params.filterByCircularity = true;
      params.minCircularity = 0.1;

      // Filter by Convexity
      params.filterByConvexity = true;
      params.minConvexity = 0.87;

      // Filter by Inertia
      params.filterByInertia = true;
      params.minInertiaRatio = 0.01;
      SimpleBlobDetector detector(params);
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
