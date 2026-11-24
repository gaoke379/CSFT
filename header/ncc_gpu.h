/**************************************************************************
* Program Name: Compressed Spatial Frequency Transfrom (CSFT)
*
* Filename: ncc_gpu.h
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

#ifndef NCC_GPU_H
#define NCC_GPU_H


#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include <npp.h> 


float* NccCuda(cv::Mat &all_descs2_expanded, cv::Mat &all_ref_subvector, int num_descriptors1, int dct_coeff_num);

#endif  // NCC_GPU_H