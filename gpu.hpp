/* 
 * The MIT License (MIT)
 *
 * Copyright (c) 2013 Michael Lancaster <mjl152@uclive.ac.nz>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to 
 * deal in the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is 
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

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
    cv::gpu::GpuMat find_face(cv::gpu::GpuMat image,
                      cv::gpu::CascadeClassifier_GPU cascade_classifier,
                       int THRESHOLD);
    cv::gpu::GpuMat blend(cv::gpu::GpuMat face1,
                          cv::gpu::GpuMat face2, double r0, double rf);
    cv::Rect get_largest_rect(cv::Rect * rects, int detections);
    cv::gpu::GpuMat to_grayscale(cv::gpu::GpuMat image);

  }
}



#endif
