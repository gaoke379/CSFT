/**************************************************************************
* Program Name: Compressed Spatial Frequency Transfrom (CSFT)
*
* Filename: ncc_cpu.h
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

#ifndef NCC_CPU_H
#define NCC_CPU_H


#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif


cv::Mat NccCpu(cv::Mat &all_descs2_expanded, cv::Mat &all_ref_subvector, int num_descriptors1, int dct_coeff_num);

#endif  // NCC_CPU_H