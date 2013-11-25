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

#include <string>
#include <iostream>
#include <sstream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "alpha_filter_kernel.h"

#include <time.h>
#include <signal.h>

#include "person.hpp"
#include "cpu.hpp"

#include "gpu.hpp"

cv::gpu::CascadeClassifier_GPU greplace::gpu::init(const char * CLASSIFIER_CONFIG,
                                                   int cuda_device) {
  cv::gpu::CascadeClassifier_GPU cascade_classifier(CLASSIFIER_CONFIG);
  cv::gpu::setDevice(0);
  return cascade_classifier;
}


cv::Rect greplace::gpu::get_largest_rect(cv::Rect * rects, int detections) {
	cv::Rect largest_rect = rects[0];
	for (unsigned int i = 1; i < detections; i ++) {
		if (rects[i].area() > largest_rect.area()) {
			largest_rect = rects[i];
		}
	}
	return largest_rect;
}


void perform_circular_alpha_filter(cv::gpu::GpuMat &x, double r0, double rf) {
  cv::gpu::PtrStepSz<uchar> ptr_step_size(x);
	alphaKernelCaller(ptr_step_size, r0, rf);
}

void perform_reverse_circular_alpha_filter(cv::gpu::GpuMat &x, double r0, double rf) {
  cv::gpu::PtrStepSz<uchar> ptr_step_size(x);
	reverseAlphaKernelCaller(ptr_step_size, r0, rf);
}

void alpha_compose(const cv::gpu::GpuMat& rgba1,
	                 const cv::gpu::GpuMat& rgba2,
                   cv::gpu::GpuMat& rgba_dest) {
	cv::Mat rgba1cpu, rgba2cpu;
	rgba1.download(rgba1cpu);
	rgba2.download(rgba2cpu);
	cv::Mat a1(rgba1.size(), rgba1.type());
	cv::Mat a2(rgba2.size(), rgba2.type());
	cv::gpu::GpuMat ra1 (rgba1.size(), rgba1.type());
	int mixch[]={3, 0, 3, 1, 3, 2, 3, 3};
	cv::mixChannels(&rgba1cpu, 1, &a1, 1, mixch, 4);
	cv::mixChannels(&rgba2cpu, 1, &a2, 1, mixch, 4);
	cv::gpu::GpuMat a1gpu, a2gpu;
	a1gpu.upload(a1);
	a2gpu.upload(a2);
  cv::gpu::PtrStepSz<uchar> ptr_1 (a1gpu);
  cv::gpu::PtrStepSz<uchar> ptr_2 (ra1);
	diff(ptr_1, ptr_2);
	cv::gpu::bitwise_or(a1gpu, cv::Scalar(0,0,0,255), a1gpu);
	cv::gpu::bitwise_or(a2gpu, cv::Scalar(0,0,0,255), a2gpu);
	cv::gpu::multiply(a2gpu, ra1, a2gpu, 1./255);
	cv::gpu::multiply(a1gpu, rgba1, a1gpu, 1./255);
	cv::gpu::multiply(a2gpu, rgba2, a2gpu, 1./255);
	cv::gpu::add(a1gpu, a2gpu, rgba_dest);
}

cv::Rect find_possible_face(cv::gpu::GpuMat greyscale,
                            cv::gpu::CascadeClassifier_GPU & haar,
                            int threshold) {
  cv::gpu::GpuMat objBuffer;
  cv::Rect ret;
  int detections = haar.detectMultiScale(greyscale, objBuffer);
  cv::Mat obj_host;
  objBuffer.colRange(0, detections).download(obj_host);
  std::cout << detections << std::endl;
  if (detections == 0) {
    ret = cv::Rect(0, 0, 0, 0);
  } else {
	  cv::Rect * possibles = obj_host.ptr<cv::Rect>();
    ret = greplace::gpu::get_largest_rect(possibles, detections);
    if (ret.area() < threshold) {
      ret = cv::Rect(0, 0, 0, 0);
    }
  }
  return ret;
}

cv::gpu::GpuMat greplace::gpu::to_grayscale(cv::gpu::GpuMat image) {
  cv::gpu::GpuMat greyscale;
  cvtColor(image, greyscale, CV_BGR2GRAY);
  return greyscale;
}

void update_image(cv::Rect face,
                                 cv::gpu::GpuMat & replacement_face,
                                 cv::gpu::GpuMat & greyscale) {
	cv::Rect faceInner (face.x + face.width*1/10,
				                    face.y + face.height*1/10, 
						                face.width * 4/5, face.height * 4/5);
  cv::gpu::GpuMat scaled_replacement_face;
	cv::gpu::resize(replacement_face, scaled_replacement_face, face.size());
	cv::Rect replacementInner (scaled_replacement_face.cols/10,
                                   scaled_replacement_face.rows/10,
				                           scaled_replacement_face.cols*4/5,
                                   scaled_replacement_face.rows*4/5);
  cv::gpu::GpuMat replacementInnerMat = scaled_replacement_face(replacementInner);
	cv::gpu::GpuMat scaledReplacementFacebgra, destROIbgra, alphaBlended;
	cv::gpu::GpuMat destROI = greyscale(faceInner);
	cvtColor(destROI, destROIbgra, CV_GRAY2BGRA);
	cvtColor(replacementInnerMat, scaledReplacementFacebgra, CV_GRAY2BGRA);
	perform_circular_alpha_filter(scaledReplacementFacebgra, 0.7, 0.9);
  perform_reverse_circular_alpha_filter(destROIbgra, 0.7, 0.9);
	alpha_compose(scaledReplacementFacebgra, destROIbgra,alphaBlended);
	cvtColor(alphaBlended, destROI, CV_RGBA2GRAY);
}

cv::gpu::GpuMat greplace::gpu::blend(cv::gpu::GpuMat face1,
                                     cv::gpu::GpuMat face2,
                                     double r0, double rf) {
  cv::gpu::GpuMat face1bgra, face2bgra, blended, final;
	cvtColor(face1, face1bgra, CV_GRAY2BGRA);
	cvtColor(face2, face2bgra, CV_GRAY2BGRA);
	perform_circular_alpha_filter(face1bgra, r0, rf);
  perform_reverse_circular_alpha_filter(face2bgra, r0, rf);
	alpha_compose(face1bgra, face2bgra, blended);
	cvtColor(blended, final, CV_RGBA2GRAY);
  return final;  
}

cv::gpu::GpuMat greplace::gpu::find_face(cv::gpu::GpuMat image,
                                        cv::gpu::CascadeClassifier_GPU
                                        cascade_classifier,
                                        int THRESHOLD) {
  cv::gpu::GpuMat objBuffer;
  cv::Rect ret;
  int detections = cascade_classifier.detectMultiScale(image, objBuffer);
  cv::Mat obj_host;
  objBuffer.colRange(0, detections).download(obj_host);
  if (detections == 0) {
    throw 0;
  } else {
	  cv::Rect * possibles = obj_host.ptr<cv::Rect>();
    ret = greplace::gpu::get_largest_rect(possibles, detections);
    if (ret.area() <= (image.rows * image.cols / THRESHOLD)) {
      throw 0;
    }
  }
  return image(ret);
}

void greplace::gpu::main_loop(cv::VideoCapture capture,
                             cv::gpu::CascadeClassifier_GPU cascade_classifier,
                             cv::Ptr<cv::FaceRecognizer> model,
                             greplace::Person previous,
                             const int THRESHOLD,
                             const int INTERPERSON_PERIOD,
                             const char * MAIN_WINDOW_TITLE) {
  greplace::Person current;
	int timeSinceLastUser = 0;
	cv::Mat image, greyscaleImage, replacementFace, scaledReplacementFace, 
      greyscaleImageBlurred, composedFace;
	cv::gpu::GpuMat imageGpu, greyscaleImageGpu, greyscaleImageBlurredGpu;
	cv::Rect previousFaceInner80;
  cv::Rect face (0, 0, 0, 0);
  cv::Rect previousFace;
	long frmCnt = 0;
	double totalT = 0.0;
	double t;
  signal(SIGINT, greplace::exit_handler);
	while (cv::waitKey(2) < 0) {
		capture >> image;
		imageGpu = cv::gpu::GpuMat(image);
		t = (double) cv::getTickCount();
		greyscaleImageGpu = greplace::gpu::to_grayscale(imageGpu);
		previousFace = face;
		face = find_possible_face(greyscaleImageGpu, cascade_classifier, THRESHOLD);
		if (face.area() != 0 && greplace::rects_overlap(face, previousFace)) {
		  /* We've detected a face */
		  /* Check if new person */
		  if (timeSinceLastUser > INTERPERSON_PERIOD) {
        previous = current;
        current.clear();
        previous.train_model(model);
		  }
		  /* Get the replacement face */
		  cv::Mat replacement = previous.prediction(image, face, model);
		  cv::gpu::GpuMat replacementGpu = cv::gpu::GpuMat(replacement);
		  update_image(face, replacementGpu, greyscaleImageGpu);
		  timeSinceLastUser = 0;
		}
    
		if (face.area() != 0) {
		  /* Add the detected face to the training list */
		  cv::Mat new_training = get_new_training_face(image, face, previous);
      current.update(new_training);
		}

		cv::GaussianBlur(greyscaleImageGpu, greyscaleImageBlurredGpu, cv::Size(9, 9), 0, 0);
		greyscaleImageBlurredGpu.download(greyscaleImageBlurred);
		cv::imshow(MAIN_WINDOW_TITLE, greyscaleImageBlurred);
		timeSinceLastUser += 50;
		t=((double)cv::getTickCount()-t)/cv::getTickFrequency();
	    totalT += t;
		frmCnt++;
		std::cout << "fps: " << 1.0/(totalT/(double)frmCnt) << std::endl;
	}
  std::cout << "greplace: error in main loop. Ending program execution." << std::endl;
  exit(EXIT_FAILURE);
}
