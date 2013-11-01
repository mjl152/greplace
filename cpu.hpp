#ifndef _GREPLACE_CPU_HPP
#define _GREPLACE_CPU_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "person.hpp"

namespace greplace {
  void main_loop(cv::VideoCapture & capture,
                 cv::CascadeClassifier cascade_classifier,
                 cv::Ptr<cv::FaceRecognizer> model,
                 greplace::Person previous,
                 const int THRESHOLD,
                 const int INTERPERSON_PERIOD,
                 const char * MAIN_WINDOW_TITLE);

  cv::Mat get_new_training_face(cv::Mat image, cv::Rect face, 
                                greplace::Person person);

  cv::CascadeClassifier init(const char * CLASSIFIER_CONFIG);

  double dist(int x, int y, int rows, int columns);

  cv::Rect get_largest_rect(cv::Vector<cv::Rect> rects);

  cv::Rect intersection(cv::Rect r1, cv::Rect r2);
  bool rects_overlap(cv::Rect r1, cv::Rect r2);
}






#endif
