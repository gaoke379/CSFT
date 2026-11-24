/**************************************************************************
* Program Name: Compressed Spatial Frequency Transfrom (CSFT)
*
* Filename: ncc_cpu.cpp
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

#include "ncc_cpu.h" 
#ifdef _OPENMP
#define OMP_MAX_THREADS 16
#endif


cv::Mat NccCpu(cv::Mat &all_descs2_expanded, cv::Mat &all_ref_subvector, int num_descriptors1, int dct_coeff_num) {
    ///
    ///-- Perform normalized cross correlation using OpenCV (CPU)
    ///   \param: all_descs2_expanded - all expanded descriptors from the matching image (image#2)
    ///   \param: all_ref_subvector - subvector template for each descriptor from the reference image (image#1)
    ///   \param: num_descriptors1 - number of descriptors from the reference image
    ///   \param: dct_coeff_num - number of DCT coefficient used for CSFT descrpitor
    ///   \return: VNCC scores
    ///
    int ref_subvector_rows = all_ref_subvector.rows / num_descriptors1;
    cv::Mat all_corr_result(num_descriptors1, all_descs2_expanded.rows-(ref_subvector_rows-1), CV_32FC1);  
#ifdef _OPENMP
    omp_set_num_threads(OMP_MAX_THREADS);
#pragma omp parallel for
#endif        
    for (int i = 0; i < num_descriptors1; i++) {       
        cv::Mat ref_subvector(all_ref_subvector(cv::Rect(0, i*ref_subvector_rows, dct_coeff_num-1, ref_subvector_rows)));  // current reference descriptor subvector
        cv::Mat corr_result(all_descs2_expanded.rows - (ref_subvector_rows - 1), 1, CV_32FC1);  // VNCC score matrix (n-by-1) for the reference descriptor
        cv::matchTemplate(all_descs2_expanded, ref_subvector, corr_result, CV_TM_CCORR_NORMED);  // compute NCC
        cv::transpose(corr_result, corr_result);  // convert to 1-by-n matrix
        cv::Mat corr_result_roi(all_corr_result(cv::Rect(0, i, all_descs2_expanded.rows-(ref_subvector_rows-1), 1)));
        corr_result.copyTo(corr_result_roi);  // copy to all_corr_result matrix
        corr_result.release();                // release memory
        corr_result_roi.release();            // release memory
        ref_subvector.release();              // release memory
    }
   
    return all_corr_result;
}