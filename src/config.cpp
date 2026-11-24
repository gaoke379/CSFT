/**************************************************************************
* Program Name: Compressed Spatial Frequency Transfrom (CSFT)
*
* Filename: config.cpp
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

#include "config.h"


ConfigParser::ConfigParser(const std::string &config_path)
{
    std::ifstream conf_file(config_path.c_str());
    if ( !conf_file.is_open() ) {
        std::cerr << "ERROR: cannot open the configuration file!" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string str_line, str_param_name, str_equal_sign;
    while(getline(conf_file, str_line)) {
        std::istringstream iss(str_line);
        iss >> str_param_name >> str_equal_sign;
        if (iss.fail() || str_equal_sign != "=" || str_param_name[0] == '#') { // skip the invalid parameters and comments in the config file
            continue;
        }
        if (str_param_name == "img1_fullpath") {
            iss >> img1_fullpath;
        }
        else if (str_param_name == "img2_fullpath") {
            iss >> img2_fullpath;
        }
        else if (str_param_name == "out_path") {
            iss >> out_path;
        }
        else if (str_param_name == "n_keypoints") {
            std::string str_flag;
            iss >> str_flag;
            n_keypoints = std::stoi(str_flag);
        }
        else if (str_param_name == "csft_desc_type") {
            iss >> csft_desc_type;
        }
        else if (str_param_name == "flag_export_keypts") {
            std::string str_flag;
            iss >> str_flag;
            flag_export_keypts = (std::stoi(str_flag) == 1)? true : false;
        }
        else if (str_param_name == "flag_export_desc") {
            std::string str_flag;
            iss >> str_flag;
            flag_export_desc = (std::stoi(str_flag) == 1)? true : false;
        }
        else if (str_param_name == "flag_export_match") {
            std::string str_flag;
            iss >> str_flag;
            flag_export_match = (std::stoi(str_flag) == 1)? true : false;
        }
        else if (str_param_name == "flag_draw_keypts_img") {
            std::string str_flag;
            iss >> str_flag;
            flag_draw_keypts_img = (std::stoi(str_flag) == 1)? true : false;
        }
        else if (str_param_name == "flag_draw_match_img") {
            std::string str_flag;
            iss >> str_flag;
            flag_draw_match_img = (std::stoi(str_flag) == 1)? true : false;
        }
        else {
            std::cerr << "ERROR: invaid input parameters!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}


ConfigParser::~ConfigParser(){}

