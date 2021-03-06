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
#include <exception>

#include <time.h>
#include <signal.h>

#include "cpu.hpp"

cv::CascadeClassifier greplace::init(const char * CLASSIFIER_CONFIG) {
  cv::CascadeClassifier cascade_classifier(CLASSIFIER_CONFIG);
  return cascade_classifier;
}

cv::Rect greplace::get_largest_rect(cv::Vector<cv::Rect> rects) {
	cv::Rect largest_rect = rects[0];
	for (size_t i = 1; i < rects.size(); i ++) {
		if (rects[i].area() > largest_rect.area()) {
			largest_rect = rects[i];
		}
	}
	return largest_rect;
}

double greplace::dist(int x, int y, int rows, int columns) {
	return sqrt(pow(abs(static_cast<double>(x) -
                      static_cast<double>(rows) / 2), 2) +
                      pow(abs(static_cast<double>(y) - 
                      static_cast<double>(columns) / 2), 2));
}

void perform_circular_alpha_filter(cv::Mat &input, double r0, double rf) {
	double max_dist = greplace::dist(0, 0, input.rows, input.cols);
  double mf = 255 / (rf - r0);
	for (int row = 0; row < input.rows; row ++ ) {
		uchar *p = input.ptr(row);
		for (int col = 0; col < input.cols; col ++) {
			int offset = 4*(col-1);
			if (offset < 0) {
				offset = 0;
			}
			double ratio = greplace::dist(row, col, input.rows, input.cols) / max_dist;
			int alpha = 255;
      if (ratio >= r0) {
        alpha = 255 - mf*(ratio - r0);
      }
      if (ratio >= rf) {
        alpha = 0;
      }
			p[offset + 3] = alpha;
		}
	}
}

void perform_reverse_circular_alpha_filter(cv::Mat &input, double r0, double rf) {
	double max_dist = greplace::dist(0, 0, input.rows, input.cols);
  double mf = 255 / (rf - r0);
	for (int row = 0; row < input.rows; row ++ ) {
		uchar *p = input.ptr(row);
		for (int col = 0; col < input.cols; col ++) {
			int offset = 4*(col-1);
			if (offset < 0) {
				offset = 0;
			}
			double ratio = greplace::dist(row, col, input.rows, input.cols) / max_dist;
			int alpha = 0;
      if (ratio >= r0) {
        alpha = mf*(ratio - r0);
      }
      if (ratio >= rf) {
        alpha = 255;
      }
			p[offset + 3] = alpha;
		}
	}
}

void alpha_compose(const cv::Mat& rgba1, const cv::Mat& rgba2,
                             cv::Mat& rgba_dest) {
	cv::Mat a1(rgba1.size(), rgba1.type()), ra1;
	cv::Mat a2(rgba2.size(), rgba2.type());
	int mixch[]={3, 0, 3, 1, 3, 2, 3, 3};
	mixChannels(&rgba1, 1, &a1, 1, mixch, 4);
	mixChannels(&rgba2, 1, &a2, 1, mixch, 4);
	subtract(cv::Scalar::all(255), a1, ra1);
	bitwise_or(a1, cv::Scalar(0,0,0,255), a1);
	bitwise_or(a2, cv::Scalar(0,0,0,255), a2);
	multiply(a2, ra1, a2, 1./255);
	multiply(a1, rgba1, a1, 1./255);
	multiply(a2, rgba2, a2, 1./255);
	add(a1, a2, rgba_dest);
}

cv::Rect greplace::intersection(cv::Rect r1, cv::Rect r2) {
  cv::Rect intersection; 

  // find overlapping region 
  intersection.x = (r1.x < r2.x) ? r2.x : r1.x; 
  intersection.y = (r1.y < r2.y) ? r2.y : r1.y; 
  intersection.width = (r1.x + r1.width < r2.x + r2.width) ? 
                        r1.x + r1.width : r2.x + r2.width; 
  intersection.width -= intersection.x; 
  intersection.height = (r1.y + r1.height < r2.y + r2.height) ? 
                         r1.y + r1.height : r2.y + r2.height; 
  intersection.height -= intersection.y; 

  // check for non-overlapping regions 
  if ((intersection.width <= 0) || (intersection.height <= 0)) { 
      intersection = cvRect(0, 0, 0, 0); 
  } 

  return intersection; 
}

cv::Rect find_possible_face(cv::Mat image,
                            cv::CascadeClassifier haar_cascade,
                            int threshold) {
  std::vector<cv::Rect> possibles;
  cv::Mat greyscale;
  cv::Rect ret;
  cvtColor(image, greyscale, CV_BGR2GRAY);
  haar_cascade.detectMultiScale(greyscale, possibles);
  if (possibles.size() == 0) {
    ret = cv::Rect(0, 0, 0, 0);
  } else {
    ret = greplace::get_largest_rect(possibles);
    if (ret.area() < threshold) {
      ret = cv::Rect(0, 0, 0, 0);
    }
  }
  return ret;
}

bool greplace::rects_overlap(cv::Rect r1, cv::Rect r2) {
  cv::Rect inter = intersection(r1, r2);
  cv::Rect sum   = r1 | r2;
  if ((static_cast<double>(inter.area()) /
      static_cast<double>(sum.area())) > 0.5) {
    return true;
  }
  return false;
}

cv::Mat to_grayscale(cv::Mat image) {
  cv::Mat greyscale;
  cvtColor(image, greyscale, CV_BGR2GRAY);
  return greyscale;
}

cv::Mat greplace::to_grayscale(cv::Mat image) {
  return to_grayscale(image);
}

cv::Mat greplace::get_new_training_face(cv::Mat image, cv::Rect face,
                                        greplace::Person person) {
	cv::Mat new_training = to_grayscale(image(face));
  cv::Mat f = person.face();
	cv::Mat new_training_resized = f.clone();
	resize(new_training, new_training_resized, f.size());
  return new_training_resized;
}


cv::Mat update_image(cv::Rect face, cv::Mat replacement_face,
                     cv::Mat greyscale, double r0, double rf) {
	cv::Rect faceInner(face.x + face.width * 1/10,
				                    face.y + face.height * 1/10, 
								            face.width * 4/5, face.height * 4/5);
  cv::Mat scaled_replacement_face;
  resize(replacement_face, scaled_replacement_face, face.size());
	cv::Rect replacementInner (scaled_replacement_face.cols / 10,
                                  scaled_replacement_face.rows / 10,
				                          scaled_replacement_face.cols * 4 / 5,
                                  scaled_replacement_face.rows * 4 / 5);
  cv::Mat replacementInnerMat = scaled_replacement_face(replacementInner);
	cv::Mat scaledReplacementFacebgra, destROIbgra, alphaBlended;
	cv::Mat destROI = greyscale(faceInner);
	cvtColor(destROI, destROIbgra, CV_GRAY2BGRA);
	cvtColor(replacementInnerMat, scaledReplacementFacebgra, CV_GRAY2BGRA);
	perform_circular_alpha_filter(scaledReplacementFacebgra, r0, rf);
	alpha_compose(scaledReplacementFacebgra, destROIbgra, alphaBlended);
	cvtColor(alphaBlended, destROI, CV_RGBA2GRAY);
  return greyscale;
}

void greplace::exit_handler(int signo) {
	std::cout << std::endl << "greplace: User entered kill signal" << std::endl;
	exit(EXIT_SUCCESS);
}

cv::Mat greplace::find_face(cv::Mat image,
                            cv::CascadeClassifier classifier,
                            int THRESHOLDING_FACTOR) {
  std::vector<cv::Rect> possibles;
  cv::Rect ret;
  classifier.detectMultiScale(image, possibles);
  if (possibles.size() == 0) {
    throw 0;
  } else {
    ret = greplace::get_largest_rect(possibles);
    if (ret.area() <= (image.rows * image.cols / THRESHOLDING_FACTOR)) {
      throw 0;
    }
  }
  return image(ret);
}

cv::Mat greplace::blend(cv::Mat face1, cv::Mat face2, double r0, double rf) {
  cv::Mat face1bgra, face2bgra, blended, final;
	cvtColor(face1, face1bgra, CV_GRAY2BGRA);
	cvtColor(face2, face2bgra, CV_GRAY2BGRA);
	perform_circular_alpha_filter(face1bgra, r0, rf);
  perform_reverse_circular_alpha_filter(face2bgra, r0, rf);
	alpha_compose(face1bgra, face2bgra, blended);
	cvtColor(blended, final, CV_RGBA2GRAY);
  return final;
}

void greplace::main_loop(cv::VideoCapture & capture,
                         cv::CascadeClassifier cascade_classifier,
                         cv::Ptr<cv::FaceRecognizer> model,
                         greplace::Person previous,
                         const int THRESHOLD,
                         const int INTERPERSON_PERIOD,
                         const char * MAIN_WINDOW_TITLE) {
  cv::Mat image, greyscale, final_image;
  cv::Rect previous_face, face;
  greplace::Person current;
  int timeSinceLastUser = 0, frmCnt = 0;
  double totalT;
  signal(SIGINT, greplace::exit_handler);
  capture.grab();
  while (cv::waitKey(2) < 0) {
    capture >> image;
    double t = static_cast<double>(cv::getTickCount());
    greyscale = to_grayscale(image);
    previous_face = face;
    face = find_possible_face(image, cascade_classifier, THRESHOLD);
    if (face.area() != 0 && rects_overlap(face, previous_face)) {
      /* We've detected a face */
      /* Check if new person */
		  if (timeSinceLastUser > INTERPERSON_PERIOD) {
        previous = current;
        current.clear();
        previous.train_model(model);
		  }
      /* Get the replacement face */
      cv::Mat replacement = previous.prediction(image, face, model);
      greyscale = update_image(face, replacement, greyscale, 0.7, 0.9);
      timeSinceLastUser = 0;
    }   
    if (face.area() != 0) {
      /* Add the detected face to the training list */
      cv::Mat new_training = get_new_training_face(image, face, previous);
      current.update(new_training);
    }

    cv::GaussianBlur(greyscale, final_image, cv::Size(9, 9), 0, 0);
    cv::imshow(MAIN_WINDOW_TITLE, final_image);
    timeSinceLastUser += 50;
    t = (static_cast<double>(cv::getTickCount())-t)/cv::getTickFrequency();
    totalT += t;
    frmCnt++;
    std::cout << "fps: " << 1.0/(totalT/(double)frmCnt) << std::endl;
  }
  std::cout << "greplace: Unexpected exit." << std::endl;
  exit(EXIT_FAILURE);
}
