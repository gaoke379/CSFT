/**************************************************************************
* Program Name: Compressed Spatial Frequency Transfrom (CSFT)
*
* Filename: csft_desc.h
*
* Copyright (C) 2022. Ke Gao, Dr. K. Palaniappan,
* and Curators of the University of Missouri, a public corporation.
* All Rights Reserved.
*
* Created by
* Ke Gao
* Department of Electrical Engineering and Computer Science
* University of Missouri-Columbia
* Columbia, MO 65211
* kegao@mail.missouri.edu
*
*
* For more information, contact:
*
* Dr. K. Palaniappan
* 205 Naka Hall (EBW)
* University of Missouri-Columbia
* Columbia, MO 65211
* palaniappank@missouri.edu
*
**************************************************************************/

#ifndef CSFT_DESC_H
#define CSFT_DESC_H

#include <stdio.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <math.h>
#include <ctime>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif


#define PI 3.14159265


struct HashPair 
{
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2>& p) const
    {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);
        return hash1 ^ hash2;
    }
};

struct SortMatchBySpatialDist
{
    bool operator()(std::pair<cv::DMatch,float> a, std::pair<cv::DMatch,float> b) 
    {
        return a.second < b.second;
    }
};


class CSFTDescriptor
{
public:
    CSFTDescriptor();
    ~CSFTDescriptor();

private:
    int max_crop_size_;                                                           // maximum crop size around a feature keypoint
    int patch_size_;                                                              // size of original local patch around feature point used to perform rotation and crop
    float rot_step_;                                                              // step size (degrees) for rotating the patch
    int num_rots_;                                                                // number of rotations for a patch
    float crop_scale_factor_;                                                     // scale factor between crop sizes
    int min_crop_size_;                                                           // minimum crop size around a feature keypoint
    int dct_coeff_num_;                                                           // number of DCT coefficient used for CSFT descrpitor
    float match_ratio_thresh_;                                                    // ratio between the best match and the second best match
    float good_match_percent_;                                                    // percentage of good feature matches based on spatial locality
    int num_crops_;                                                               // number of crops for a patch
    std::vector<int> crop_sizes_;                                                 // size of different crops
    std::unordered_map<int, std::pair<int, int>> dct_coeff_idx_to_loc_;           // mapping of indices of DCT coefficients in zigzag order to their (x,y) locations in DCT matrix
    std::unordered_map<std::pair<int, int>, int, HashPair> dct_coeff_loc_to_idx_; // mapping of DCT coefficient (x,y) locations in DCT matrix to their indices in zigzag order 

private:
    cv::Mat ImRotate(cv::Mat &img, float &angle, float &scale);
    void IndexZigzagScan(int &rows, int &cols);
    std::vector<float> ZigzagScan(cv::Mat &img);
    cv::Mat NormMat(cv::Mat &in_mat);   
    cv::Mat Rotate90DCT(const cv::Mat &dct_coeffs);
    cv::Mat Rotate180DCT(const cv::Mat &dct_coeffs);
    cv::Mat Rotate270DCT(const cv::Mat &dct_coeffs);
    cv::Mat DescriptorExpand(cv::Mat &desc_orig);
    std::vector<cv::DMatch> CleanMatches(std::vector<cv::DMatch> matches_mto1);
    std::vector<cv::DMatch> RefineMatches(std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> matches);

public:
    cv::Mat DescriptorExtract(cv::Mat &img_gray, std::vector<cv::KeyPoint> &keypoints, std::string csft_desc_type);
    std::vector<cv::DMatch> KNNMatcher(cv::Mat &descriptors1, cv::Mat &descriptors2);
    std::vector<cv::DMatch> VNCCMatcher(std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2, cv::Mat &descriptors1, cv::Mat &descriptors2);
};

#endif // CSFT_DESC_H