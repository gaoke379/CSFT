/**
 * @file utils.h
 * @brief Some utilities functions
 * @date Oct 07, 2014
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#pragma once

/* ************************************************************************* */
// OpenCV
#include <opencv2/features2d/features2d.hpp>

// System
#include <vector>
#include <iostream>
#include <iomanip>

/* ************************************************************************* */
// Stringify common types such as int, double and others.
template <typename T>
inline std::string to_string(const T& x) {
  std::stringstream oss;
  oss << x;
  return oss.str();
}

/* ************************************************************************* */
// Stringify and format integral types as follows:
// to_formatted_string(  1, 2) produces string:  '01'
// to_formatted_string(  5, 2) produces string:  '05'
// to_formatted_string( 19, 2) produces string:  '19'
// to_formatted_string( 19, 3) produces string: '019'
template <typename Integer>
inline std::string to_formatted_string(Integer x, int num_digits) {
  std::stringstream oss;
  oss << std::setfill('0') << std::setw(num_digits) << x;
  return oss.str();
}

/* ************************************************************************* */
/// Compute the minimum value of a float image
void compute_min_32F(const cv::Mat& src, float& value);

/// Compute the maximum value of a float image
void compute_max_32F(const cv::Mat& src, float& value);

/// Convert the scale of the input image prior to visualization
void convert_scale(cv::Mat& src);

/// This function copies the input image and converts the scale of the copied image prior visualization
void copy_and_convert_scale(const cv::Mat& src, cv::Mat& dst);

/// This function draws the list of detected keypoints
void draw_keypoints(cv::Mat& img, const std::vector<cv::KeyPoint>& kpts);
