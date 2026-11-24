/**************************************************************************
* Program Name: Compressed Spatial Frequency Transfrom (CSFT)
*
* Filename: config.h
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

#ifndef CONFIG_H
#define CONFIG_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <queue>
#include <opencv2/opencv.hpp>


class ConfigParser
{
public:
    ConfigParser(const std::string &config_path);
    ~ConfigParser();

    std::string img1_fullpath;
    std::string img2_fullpath;
    std::string out_path;
    int n_keypoints;
    std::string csft_desc_type;
    bool flag_export_keypts;
    bool flag_export_desc;
    bool flag_export_match;
    bool flag_draw_keypts_img;
    bool flag_draw_match_img;
};

#endif  // CONFIG_H
