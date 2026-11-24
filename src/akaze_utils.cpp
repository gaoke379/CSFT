//=============================================================================
//
// utils.cpp
// Authors: Pablo F. Alcantarilla, Jesus Nuevo (2)
// Institutions: TrueVision Solutions (2)
//
// Date: 07/10/2014
// Email: pablofdezalc@gmail.com
//
// AKAZE Features Copyright 2014, Pablo F. Alcantarilla, Jesus Nuevo
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file utils.cpp
 * @brief Some utilities functions
 * @date Oct 07, 2014
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#include "akaze_utils.h"

// OpenCV
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

// System
#include <fstream>

using namespace std;

/* ************************************************************************* */
void compute_min_32F(const cv::Mat &src, float& value) {

  float aux = 1000.0;
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      if (src.at<float>(i,j) < aux)
        aux = src.at<float>(i,j);
    }
  }
  value = aux;
}

/* ************************************************************************* */
void compute_max_32F(const cv::Mat &src, float& value) {

  float aux = 0.0;
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      if (src.at<float>(i,j) > aux)
        aux = src.at<float>(i,j);
    }
  }
  value = aux;
}

/* ************************************************************************* */
void convert_scale(cv::Mat& src) {

  float min_val = 0, max_val = 0;
  compute_min_32F(src,min_val);
  src = src - min_val;
  compute_max_32F(src,max_val);
  src = src / max_val;
}

/* ************************************************************************* */
void copy_and_convert_scale(const cv::Mat &src, cv::Mat dst) {

  float min_val = 0, max_val = 0;
  src.copyTo(dst);
  compute_min_32F(dst,min_val);
  dst = dst - min_val;
  compute_max_32F(dst,max_val);
  dst = dst / max_val;
}

/* ************************************************************************* */
void draw_keypoints(cv::Mat& img, const std::vector<cv::KeyPoint>& kpts) {

  int x = 0, y = 0;
  float radius = 0.0;

  for (size_t i = 0; i < kpts.size(); i++) {
    x = (int)(kpts[i].pt.x+.5);
    y = (int)(kpts[i].pt.y+.5);
    radius = kpts[i].size/2.0;
    cv::circle(img, cv::Point(x,y), radius*2.50, cv::Scalar(0,255,0), 1);
    cv::circle(img, cv::Point(x,y), 1.0, cv::Scalar(0,0,255), -1);
  }
}
