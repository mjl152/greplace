#ifndef _GREPLACE_PERSON_HPP
#define _GREPLACE_PERSON_HPP

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

namespace greplace {
  class Person {
  public:
    Person(void);
    Person(std::string loading_directory, int x_res, int y_res);
    void train_model(cv::Ptr<cv::FaceRecognizer> model);
    void clear(void);
    void update(cv::Mat face);
    cv::Mat prediction(cv::Mat image, cv::Rect face, cv::Ptr<cv::FaceRecognizer> model);
    cv::Mat face(void);
  private:
    void load_training_faces(std::string loading_directory, int x_res, int y_res);
    void label_training_faces(void);
    cv::vector<cv::Mat> faces;
    std::vector<int> labels;
  };

}






#endif
