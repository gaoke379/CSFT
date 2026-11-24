/**************************************************************************
* Program Name: Compressed Spatial Frequency Transfrom (CSFT)
*
* Filename: csft_driver.cpp
*
* Description:
* C++ driver program for CSFT 
*
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


#include <stdio.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "config.h"
#include "csft_det.h"
#include "csft_desc.h"


int main(int argc, char** argv)
{
    std::string config_path; // path to the config file
    if (argc != 2) {
        std::cerr << "USAGE: ./CSFT <config_file_path>" << std::endl;
        return EXIT_FAILURE;
    }
    else {
        config_path = argv[1];
    }


    ///-- Initialization
    ///
    ConfigParser param(config_path);                        // load variables from config file
    std::string img1_fullpath = param.img1_fullpath;        // path for 1st input image
    std::string img2_fullpath = param.img2_fullpath;        // path for 2nd input image
    std::string out_path = param.out_path;                  // path for output folder
    DetectorOptions options;                                // detector options for CSFT
    options.n_keypoints = param.n_keypoints;                // number of CSFT keypoints
    std::string csft_desc_type = param.csft_desc_type;      // CSFT descriptor type (Upright or Standard); Upright-CSFT disables rotation invariance
    bool flag_export_keypts = param.flag_export_keypts;     // export feature keypoints to text files
    bool flag_export_desc = param.flag_export_desc;         // export feature descriptors to text files
    bool flag_export_match = param.flag_export_match;       // export feature matches to text files
    bool flag_draw_keypts_img = param.flag_draw_keypts_img; // draw feature keypoints on the input images
    bool flag_draw_match_img = param.flag_draw_match_img;   // draw feature matches
    std::cout << "/***************************/" << std::endl;
    std::cout << "/** CSFT Feature Matching **/" << std::endl;
    std::cout << "/***************************/" << std::endl << std::endl;
    std::cout << "Reference image: " << img1_fullpath << std::endl;
    std::cout << "Matching image:  " << img2_fullpath << std::endl << std::endl;


    ///-- Load input images
    ///
    size_t pos_last_slash_img1 = img1_fullpath.find_last_of("/");
    std::string img1_name_ext = img1_fullpath.substr(pos_last_slash_img1+1);
    size_t pos_last_slash_img2 = img2_fullpath.find_last_of("/");
    std::string img2_name_ext = img2_fullpath.substr(pos_last_slash_img2+1);
    std::string img1_name = img1_name_ext.substr(0, img1_name_ext.length()-4);
    std::string img2_name = img2_name_ext.substr(0, img2_name_ext.length()-4);
    cv::Mat img1 = cv::imread(img1_fullpath, cv::IMREAD_COLOR); // read the first image as the query image
    cv::Mat img2 = cv::imread(img2_fullpath, cv::IMREAD_COLOR); // read the second image as the train image
    if (!img1.data || !img2.data) { // check whether the image is loaded or not
        std::cerr << "ERROR: Images cannot be loaded!!!" << std::endl;
        return EXIT_FAILURE;
    }

    //-- Convert to grayscale images if inputs are RGB
    cv::Mat img1_gray, img2_gray;
    if (img1.channels() == 3) {
        cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY); }
    else {
        img1_gray = img1.clone(); }
    if (img2.channels() == 3) {
        cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY); }
    else {
        img2_gray = img2.clone(); 
    }


    ///-- Extract CSFT feature keypoints
    ///
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat img1_gray_32f, img2_gray_32f;
    img1_gray.convertTo(img1_gray_32f, CV_32F, 1.0/255.0, 0);
    img2_gray.convertTo(img2_gray_32f, CV_32F, 1.0/255.0, 0);
    options.img_width = img1.cols;
    options.img_height = img1.rows;
    CSFTDetector detector1(options);
    options.img_width = img2.cols;
    options.img_height = img2.rows;
    CSFTDetector detector2(options);
    detector1.Create_Nonlinear_Scale_Space(img1_gray_32f);
    detector1.Feature_Detection(keypoints1);
    detector2.Create_Nonlinear_Scale_Space(img2_gray_32f);
    detector2.Feature_Detection(keypoints2);
    std::cout << "Number of keypoints in image " << img1_name_ext << ": " << keypoints1.size() << std::endl;
    std::cout << "Number of keypoints in image " << img2_name_ext << ": " << keypoints2.size() << std::endl;
    
    //-- Export feature keypoints to txt files
    if (flag_export_keypts) {
        FILE *datafile_ft1, *datafile_ft2;
        std::string save_name1 = out_path + img1_name + "_csft_keypts.txt";
        std::string save_name2 = out_path + img2_name + "_csft_keypts.txt";
        datafile_ft1 = fopen(save_name1.c_str(), "w");
        datafile_ft2 = fopen(save_name2.c_str(), "w");
        int num_points = keypoints1.size();
        for (int i = 0; i < num_points; i++) {
            fprintf(datafile_ft1, "%.6f %.6f\n", keypoints1[i].pt.x, keypoints1[i].pt.y); // write to txt file
        }
        num_points = keypoints2.size();
        for (int i = 0; i < num_points; i++) {
            fprintf(datafile_ft2, "%.6f %.6f\n", keypoints2[i].pt.x, keypoints2[i].pt.y); // write to txt file
        }
        fclose(datafile_ft1);
        fclose(datafile_ft2);
        std::cout << "Keypoints text files save to:           " << out_path << std::endl;
    }

    //-- Draw keypoints
    if (flag_draw_keypts_img) {
        cv::Mat img_keypoints1, img_keypoints2;
        cv::drawKeypoints(img1, keypoints1, img_keypoints1, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DEFAULT);
        cv::drawKeypoints(img2, keypoints2, img_keypoints2, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DEFAULT);
        cv::imwrite(out_path+img1_name+"_csft.png", img_keypoints1);
        cv::imwrite(out_path+img2_name+"_csft.png", img_keypoints2);
        std::cout << "Keypoints visualization images save to: " << out_path << std::endl;
    }


    ///-- Extract descriptor for each feature point
    ///
    CSFTDescriptor csft;
    cv::Mat descriptors1 = csft.DescriptorExtract(img1_gray, keypoints1, csft_desc_type);
    cv::Mat descriptors2 = csft.DescriptorExtract(img2_gray, keypoints2, csft_desc_type);
    std::cout << std::endl << "Feature descriptors generated" << std::endl;

    //-- Export feature descriptors to txt files
    if (flag_export_desc) {
        FILE *datafile_desc1, *datafile_desc2;
        std::string save_name1 = out_path + img1_name + "_csft_desc.txt";
        std::string save_name2 = out_path + img2_name + "_csft_desc.txt";
        datafile_desc1 = fopen(save_name1.c_str(), "w");
        for (int i = 0; i < descriptors1.rows; i++) {
            for (int j = 0; j < descriptors1.cols; j++) {
                fprintf(datafile_desc1, "%.6f ", descriptors1.at<float>(i, j));
            }
            fprintf(datafile_desc1, "\n");
        }
        datafile_desc2 = fopen(save_name2.c_str(), "w");
        for (int i = 0; i < descriptors2.rows; i++) {
            for (int j = 0; j < descriptors2.cols; j++) {
                fprintf(datafile_desc2, "%.6f ", descriptors2.at<float>(i, j));
            }
            fprintf(datafile_desc2, "\n");
        }
        fclose(datafile_desc1);
        fclose(datafile_desc2);
        std::cout << "Descriptors text files save to:         " << out_path << std::endl;       
    }


    ///-- Feature matching
    ///
    std::cout << std::endl << "Matching features..." << std::endl;
    std::vector<cv::DMatch> matches_good;
    if (csft_desc_type == "Upright") {  // use standard distance ratio matching 
        matches_good = csft.KNNMatcher(descriptors1, descriptors2);      
    }
    else if (csft_desc_type == "Standard") {  // use VNCC + distance ratio matching
        matches_good = csft.VNCCMatcher(keypoints1, keypoints2, descriptors1, descriptors2);
    }
    else {
        std::cerr << "ERROR: Invalid CSFT descriptor type!!!" << std::endl;
        return EXIT_FAILURE;
    }
    int num_matches = matches_good.size();
    std::cout << std::endl << "Number of matches: " << num_matches << std::endl;
    descriptors1.release();  // release memory
    descriptors2.release();  // release memory

    //-- Export feature matches to a txt file
    FILE *datafile_match;
    std::string save_name_match = out_path + img1_name + "_" + img2_name + "_csft_matches.txt";
    if (flag_export_match) {
        datafile_match = fopen(save_name_match.c_str(), "w");
        for (size_t i = 0; i < matches_good.size(); i++) {
            cv::KeyPoint pt1 = keypoints1[matches_good[i].queryIdx];
            cv::KeyPoint pt2 = keypoints2[matches_good[i].trainIdx];
            fprintf(datafile_match, "%.6f %.6f %.6f %.6f\n", pt1.pt.x, pt1.pt.y, pt2.pt.x, pt2.pt.y); // write to txt file
        }
        fclose(datafile_match);
        std::cout << "Matches text file save to:              " << out_path << std::endl;       
    }

    //-- Draw feature matches on input images
    if (flag_draw_match_img) {
        cv::Mat img_matches;
        cv::drawMatches(img1, keypoints1, img2, keypoints2, matches_good, img_matches,
                         cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                         cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imwrite(out_path+img1_name+"_"+img2_name+"_csft.png", img_matches);
        std::cout << "Matches visualization image save to:    " << out_path << std::endl;
    }


    return EXIT_SUCCESS;
}
