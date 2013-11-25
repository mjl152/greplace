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

#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <string>
#include <iostream>
#include <sstream>
#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <dirent.h>
#include <getopt.h>

#include "person.hpp"
#include "cpu.hpp"
#include "gpu.hpp"
#include "greplace-psearch-cpu.hpp"
#include "greplace-psearch-gpu.hpp"
#include "cmake_config.h"

const int THRESHOLD = 16;


double greplace::gpu::find_statistic(std::vector<std::string> images, double r0, double rf,
                      cv::gpu::CascadeClassifier_GPU & classifier) {
  std::vector<double> statistic;
  for (size_t i = 0; i < images.size(); i ++) {
    std::string image1_location = images[i];
    for (size_t j = 0; j < images.size(); j ++) {
      std::string image2_location = images[j];
		  cv::gpu::GpuMat image1(cv::imread(image1_location,
                                        CV_LOAD_IMAGE_GRAYSCALE));
		  cv::gpu::GpuMat image2(cv::imread(image2_location,
                                        CV_LOAD_IMAGE_GRAYSCALE));
      try {
        cv::gpu::GpuMat face1 = greplace::gpu::find_face(image1, classifier,
                                                         THRESHOLD);
        cv::gpu::GpuMat face2 = greplace::gpu::find_face(image2, classifier,
                                                         THRESHOLD);
        cv::Mat face1_cpu, face2_cpu, face3_cpu;
        face1.download(face1_cpu);
        face2.download(face2_cpu);
        cv::resize(face2_cpu, face2_cpu, face1_cpu.size());
        face2.upload(face2_cpu);
        cv::gpu::GpuMat face3 = greplace::gpu::blend(face1, face2, r0, rf);
        face3.download(face3_cpu);
        cv::imshow("face1", face1_cpu);
        cv::imshow("face2", face2_cpu);
        cv::imshow("Blended", face3_cpu);
        cv::waitKey(10);
        cv::Mat hist1 = greplace::hist(face1_cpu);
        cv::Mat hist2 = greplace::hist(face2_cpu);
        cv::Mat hist3 = greplace::hist(face3_cpu);
        double s = greplace::hist_correlation(hist1, hist3) + 
                   greplace::hist_correlation(hist2, hist3);
        statistic.push_back(s);
      }
      catch (...) {
        // One of the faces couldn't be found because the angle of the image
        // is too great for the frontal face classifier
      }
    }
  }
  return greplace::mean(statistic);
}

