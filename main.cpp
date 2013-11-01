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
#ifdef HAVE_CUDA
  #include <opencv2/gpu/gpu.hpp>
#endif

#include <string>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdint>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#include <getopt.h>

#include "person.hpp"
#include "cpu.hpp"
#include "cmake_config.h"

#ifdef HAVE_CUDA
  #include "gpu.hpp"
#endif

static const char *optString = "x:y:w:g:chv";

static const struct option longOpts[] = {
  {"x_res",       required_argument, NULL, 'x'},
  {"y_res",       required_argument, NULL, 'y'},
  {"webcam",      required_argument, NULL, 'w'},
  {"cuda_device", required_argument, NULL, 'g'},
  {"cpu",         no_argument,       NULL, 'c'},
  {"help",        no_argument,       NULL, 'h'},
  {"usage",       no_argument,       NULL, 'h'},
  {"verbose",     no_argument,       NULL, 'v'}
};

const int THRESHOLDING_FACTOR = 16;
const int INTERPERSON_PERIOD  = 1000;

const char * FACES_LOAD_DIRECTORY = "\\starting_faces";
const char * HAAR_CASCADE_FRONTAL_FACE_LOCATION = "haarcascade_frontalface_default.xml";
const char * MAIN_WINDOW_TITLE = "greplace";

void display_help(void) {
  std::cout << "greplace, an optionally CUDA accelerated facial detection and";
  std::cout << " replacement program."                            << std::endl;
  std::cout << "usage: greplace [options]"                        << std::endl;
  std::cout << "Available options:"                               << std::endl;
  std::cout << "    -x, --x_res"                                  << std::endl;
  std::cout << "        Sets the x resolution of the camera."     << std::endl;
  std::cout << "        Defaults to 720 pixels."                  << std::endl;
  std::cout << "    -y, --y_res"                                  << std::endl;
  std::cout << "        Sets the y resolution of the camera."     << std::endl;
  std::cout << "        Defaults to 1280 pixels."                 << std::endl;
  std::cout << "    -w, --webcam"                                 << std::endl;
  std::cout << "        Sets the webcam to use."                  << std::endl;
  std::cout << "        Defaults to 0."                           << std::endl;
  std::cout << "    -g, --cuda_device"                            << std::endl;
  std::cout << "        Sets the CUDA device to use."             << std::endl;
  std::cout << "        Defaults to 0."                           << std::endl;
  std::cout << "    -c, --cpu"                                    << std::endl;
  std::cout << "        Runs greplace on the CPU. If greplace was compiled ";
  std::cout << "without CUDA, greplace is always run on the CPU." << std::endl;
  std::cout << "    -v, --verbose"                                << std::endl;
  std::cout << "        Makes greplace output additional ";
  std::cout << "information."                                     << std::endl;
  std::cout << "    -h, --help, --usage"                          << std::endl;
  std::cout << "        Displays this help text."                 << std::endl;
  exit(EXIT_SUCCESS);
}

void get_options(int argc, char ** argv, int & x_res, int & y_res,
                 int & video_capture, int & cuda_device, bool & gpu,
                 bool & verbosity) {
  int optIndex[1];
  int opt;

  while ((opt = getopt_long(argc, argv, optString, longOpts, optIndex)) != -1) {
    switch (opt) {
    case 0:
      display_help();
    case 'x':
			x_res = atoi(optarg);
      break;
    case 'y':
			y_res = atoi(optarg);
      break;
    case 'c':
      gpu = false;
      break;
    case 'g':
			cuda_device = atoi(optarg);
      break;
    case 'v':
      verbosity = true;
      break;
    case 'w':
			video_capture = atoi(optarg);
      break;
    case 'h':
      display_help();
    default:
      display_help();
    }
  }
}


int main(int argc, char ** argv) {
  int x_res = 1280, y_res = 720, video_capture = 0, cuda_device = 0, threshold;
  bool verbose = false, gpu = true;
  get_options(argc, argv, x_res, y_res, video_capture, cuda_device, gpu,
              verbose);
  if ((HAVE_CUDA == false) && (gpu = true)) {
    std::cout << "greplace was compiled without CUDA support. Proceeding on ";
    std::cout << "CPU." << std::endl;
  }
  threshold = x_res * y_res / THRESHOLDING_FACTOR;
  cv::VideoCapture webcam(video_capture);
	webcam.set(CV_CAP_PROP_FRAME_WIDTH,  x_res);
	webcam.set(CV_CAP_PROP_FRAME_HEIGHT, y_res);
  cv::namedWindow(MAIN_WINDOW_TITLE, CV_WINDOW_AUTOSIZE );
  auto model = cv::createFisherFaceRecognizer();
  auto previous_person = greplace::Person(std::string(FACES_LOAD_DIRECTORY),
                                          x_res, y_res);
  previous_person.train_model(model);
  #ifdef HAVE_CUDA
  if (gpu) {
    auto classifier = greplace::gpu::init(HAAR_CASCADE_FRONTAL_FACE_LOCATION,
                                          cuda_device);
    greplace::gpu::main_loop(webcam, classifier, model, previous_person,
                             threshold, INTERPERSON_PERIOD,
                             MAIN_WINDOW_TITLE);
  } else {
  #endif
    auto classifier = greplace::init(HAAR_CASCADE_FRONTAL_FACE_LOCATION);
    greplace::main_loop(webcam, classifier, model, previous_person, threshold,
                        INTERPERSON_PERIOD, MAIN_WINDOW_TITLE);
  #ifdef HAVE_CUDA
  }
  #endif

  return EXIT_FAILURE;
}
