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
#include <math.h>

#include "person.hpp"
#include "greplace-psearch-cpu.hpp"
#include "cpu.hpp"
#include "cmake_config.h"

static const char *optString = "s:t:e:f:m:n";
const int THRESHOLD = 16;

double greplace::mean(std::vector<double> set) {
	double m = 0;
	for (size_t i = 0; i < set.size(); i ++) {
    m += set[i];
  }
  m /= set.size();
  return m;
}

double standard_deviation(std::vector <double> set) {
  double m = greplace::mean(set);
  double s = 0;
  for (auto item : set) {
    s += pow(item - m, 2.0);
  }
  s /= set.size();
  return sqrt(s);
}

double greplace::hist_correlation(cv::Mat H1, cv::Mat H2) {
  return cv::compareHist(H1, H2, CV_COMP_CORREL);
}

cv::Mat greplace::hist(cv::Mat const & image) {
	int bins = 256;
  int histSize[] = {bins};
  float lranges[] = {0, 256};
  const float * ranges[] = {lranges};
  int channels[] = {0};
  cv::Mat hist;
  cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true,
               false);
  return hist;
}

std::vector<cv::Mat> greplace::hists(std::vector<cv::Mat> & faces) {
  std::vector<cv::Mat> s;
  for (auto face : faces) {
    s.push_back(greplace::hist(face));
  }
  return s;
}

double greplace::find_statistic(std::vector<cv::Mat> images, std::vector<cv::Mat> hists, double r0,
                               double rf, cv::CascadeClassifier & classifier,
                               double & std_d) {
  std::vector<double> statistic;
  for (size_t i = 0; i < images.size(); i ++) {
      auto face1 = images[i];
      auto hist1 = hists[i];
		  for (size_t j = 0; j < images.size(); j ++) {
          auto face2 = images[j];
          auto hist2 = hists[j];
		      cv::resize(face2, face2, face1.size());
		      cv::Mat face3 = greplace::blend(face1, face2, r0, rf);
          cv::imshow("Host", face1);
          cv::imshow("Replacement", face2);
          cv::imshow("Blended", face3);
          cv::waitKey(3000);
		      cv::Mat hist3 = hist(face3);
		      double s = hist_correlation(hist1, hist3) + hist_correlation(hist2, hist3);
		      statistic.push_back(s);
		  }
  }
  std_d = standard_deviation(statistic);
  return mean(statistic);
}
