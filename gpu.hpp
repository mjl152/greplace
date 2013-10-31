#ifndef _GREPLACE_GPU_HPP
#define _GREPLACE_GPU_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "person.hpp"

namespace greplace {
  namespace gpu {
    void main_loop(cv::VideoCapture capture,
                   cv::gpu::CascadeClassifier_GPU cascade_classifier,
                   cv::Ptr<cv::FaceRecognizer> model,
                   greplace::Person previous,
                   const int THRESHOLD,
                   const int INTERPERSON_PERIOD,
                   const char * MAIN_WINDOW_TITLE);

    
    cv::gpu::CascadeClassifier_GPU init(const char * CLASSIFIER_CONFIG,
                                        int cuda_device);

  }
}



#endif
