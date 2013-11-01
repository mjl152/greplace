#include <string>
#include <vector>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "person.hpp"

greplace::Person::Person(void) { };

greplace::Person::Person(std::string load_directory, int x_res, int y_res) {
  load_training_faces(load_directory, x_res, y_res);
  label_training_faces();
}

cv::Mat greplace::Person::prediction(cv::Mat image, cv::Rect face, cv::Ptr<cv::FaceRecognizer> model) {
  cv::Mat greyscale;
  cvtColor(image, greyscale, CV_BGR2GRAY);
  auto faceMat = greyscale(face);
  auto replacement = faces[0].clone();
  resize(faceMat, replacement, replacement.size());
  auto result = model->predict(replacement);
  return faces[result];
}

void greplace::Person::load_training_faces(std::string load_directory, int x_res, int y_res) {
	int i;
  std::cout << load_directory << std::endl;
	for (i = 1; i <= 10; i ++) {
		std::ostringstream out;
    out << i << ".pgm";
		cv::Mat loaded = cv::imread(out.str(), 0);
		cv::Mat scaled;
		resize(loaded, scaled, cv::Size(x_res/4, y_res/4));
		faces.push_back(scaled);
	}
}

void greplace::Person::label_training_faces(void) {
	for (size_t i = 0; i < faces.size(); i ++) {
		labels.push_back(i);
	}
}

void greplace::Person::train_model(cv::Ptr<cv::FaceRecognizer> model) {
  model->train(faces, labels);
}

void greplace::Person::update(cv::Mat face) {
  faces.push_back(face);
 	if (faces.size() > 15) {
		faces.erase(faces.begin() + 1);
	}
	if (labels.size() == 0) {
		labels.push_back(0);
	} else {
		labels.push_back(labels[labels.size() - 1] + 1);
	}
	if (labels.size() > 15) {
		labels.pop_back();
	}
}

void greplace::Person::clear(void) {
  faces.clear();
  labels.clear();
}

cv::Mat greplace::Person::face(void) {
  return faces[0];
}
