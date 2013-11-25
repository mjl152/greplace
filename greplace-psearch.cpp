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
#ifdef HAVE_GPU
  #include <opencv2/gpu/gpu.hpp>
#endif

#include <string>
#include <iostream>
#include <sstream>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <dirent.h>
#include <getopt.h>

#include "greplace-psearch-cpu.hpp"
#ifdef HAVE_GPU
#include "gpu.hpp"
#include "greplace-psearch-gpu.hpp"
#endif
#include "person.hpp"
#include "cpu.hpp"
#include "cmake_config.h"

static const char *optString = "s:t:e:f:m:n";
const int THRESHOLD = 16;

static const char *IMAGE_DIR = "psearch_images";
const char * HAAR_CASCADE_FRONTAL_FACE_LOCATION = "haarcascade_frontalface_default.xml";

static const struct option longOpts[] = {
  {"rf0",         required_argument, NULL, 's'},
  {"rff",         required_argument, NULL, 't'},
  {"r00",         required_argument, NULL, 'e'},
  {"r0f",         required_argument, NULL, 'f'},
  {"delta_r0",    required_argument, NULL, 'm'},
  {"delta_rf",    required_argument, NULL, 'n'},
  {"cpu",         no_argument,       NULL, 'c'},
  {"help",        no_argument,       NULL, 'h'},
  {"usage",       no_argument,       NULL, 'h'},
  {"verbose",     no_argument,       NULL, 'v'}
};

void display_help(void) {
  std::cout << "greplace-psearch, a program to find the optimal radial ";
  std::cout << "alpha filter for face blending."                  << std::endl;
  std::cout << "usage: greplace-psearch [options]"                << std::endl;
  std::cout << "Available options:"                               << std::endl;
  std::cout << "    -s, --rf0"                                    << std::endl;
  std::cout << "        Sets the starting value of rf."           << std::endl;
  std::cout << "        Defaults to 0."                           << std::endl;
  std::cout << "    -t, --rff"                                    << std::endl;
  std::cout << "        Sets the final value of rf."              << std::endl;
  std::cout << "        Defaults to 1."                           << std::endl;
  std::cout << "    -e, --r00"                                    << std::endl;
  std::cout << "        Sets the starting value of r0."           << std::endl;
  std::cout << "        Defaults to 0."                           << std::endl;
  std::cout << "    -f, --r0f"                                    << std::endl;
  std::cout << "        Sets the final value of r0."              << std::endl;
  std::cout << "        Defaults to 1."                           << std::endl;
  std::cout << "    -m, --delta_r0"                               << std::endl;
  std::cout << "        Sets the step size for r0. Defaults to 0.1";
  std::cout << std::endl;
  std::cout << "    -n, --delta_rf"                               << std::endl;
  std::cout << "        Sets the step size for rf. Defaults to 0.1";
  std::cout << std::endl;
  std::cout << "    -v, --verbose"                                << std::endl;
  std::cout << "        Makes greplace-psearch output additional ";
  std::cout << "information."                                     << std::endl;
  std::cout << "    -h, --help, --usage"                          << std::endl;
  std::cout << "        Displays this help text."                 << std::endl;
  exit(EXIT_SUCCESS);
}

void get_options(int argc, char ** argv, double & r00, double & r0f,
                 double & rf0, double & rff, double & delta_r0,
                 double & delta_rf, bool & verbosity) {
  int optIndex[1];
  int opt;
  while ((opt = getopt_long(argc, argv, optString, longOpts, optIndex)) != -1) {
    switch (opt) {
    case 0:
      display_help();
    case 's':
			rf0 = atof(optarg);
      break;
    case 't':
			rff = atof(optarg);
      break;
    case 'e':
      r00 = atof(optarg);
      break;
    case 'f':
			r0f = atof(optarg);
      break;
    case 'm':
      delta_r0 = atof(optarg);
      break;
    case 'n':
			delta_rf = atof(optarg);
      break;
    case 'v':
      verbosity = true;
      break;
    case 'h':
      display_help();
    default:
      display_help();
    }
  }
}

std::vector<std::string> test_files (const char * dir_name) {
	std::vector<std::string> v;
  DIR * dir;
  std::string dir_string(dir_name);
  struct dirent *ent;
  if ((dir = opendir(dir_name)) != NULL) {
    while ((ent = readdir(dir)) != NULL) {
      std::string s(ent->d_name);
      if (s.find(".jpg") != std::string::npos) {
        std::stringstream ss;
				ss << dir_string << "/" << s;
      	v.push_back(ss.str());
			}
    }
    closedir(dir);
  }
  return v;
}

int main(int argc, char ** argv) {
	double r00 = 0.6, r0f = 1, rf0 = 0.8, rff = 1, delta_r0 = 0.05, delta_rf = 0.05;
	bool verbose = false;
  get_options(argc, argv, r00, r0f, rf0, rff, delta_r0, delta_rf, verbose);
  std::vector<std::string> images = test_files(IMAGE_DIR);
  std::vector<cv::Mat> images_mat;
  for (auto i : images) {
    try {
      auto image = cv::imread(i, CV_LOAD_IMAGE_GRAYSCALE);
      images_mat.push_back(image);
    }
    catch (...) {

    }
  }
  #ifndef HAVE_GPU
  cv::CascadeClassifier classifier = greplace::init(
                                            HAAR_CASCADE_FRONTAL_FACE_LOCATION);
  #else
  cv::gpu::CascadeClassifier_GPU classifier = greplace::gpu::init(HAAR_CASCADE_FRONTAL_FACE_LOCATION, 0);
  #endif
  std::vector<cv::Mat> faces;
  for (auto i : images_mat) {
    try {
      auto face = greplace::find_face(i, classifier, 16);
      faces.push_back(face);
    } catch (...) {

    }
  }
  double std_d;
  std::vector<cv::Mat> hists = greplace::hists(faces);
  cv::namedWindow("Host");
  cv::namedWindow("Replacement");
  cv::namedWindow("Blended");
  for (double r0 = r00; r0 <= r0f; r0 += delta_r0) {
    for (double rf = rf0; rf <= rff; rf += delta_rf) {
#ifndef HAVE_GPU
			double s = greplace::find_statistic(faces, hists, r0, rf, classifier, std_d);
#else
      double s = greplace::gpu::find_statistic(faces, hists, r0, rf, classifier);
#endif
			std::cout << r0 << ", " << rf << ", " << s << ", " << std_d << std::endl;
    }
  }
}
