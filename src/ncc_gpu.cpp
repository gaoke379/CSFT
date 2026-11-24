/**************************************************************************
* Program Name: Compressed Spatial Frequency Transfrom (CSFT)
*
* Filename: ncc_gpu.cpp
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

#include "ncc_gpu.h" 


float* NccCuda(cv::Mat &all_descs2_expanded, cv::Mat &all_ref_subvector, int num_descriptors1, int dct_coeff_num) {
    ///
    ///-- Perform normalized cross correlation using CUDA
    ///   \param: all_descs2_expanded - all expanded descriptors from the matching image (image#2)
    ///   \param: all_ref_subvector - subvector template for each descriptor from the reference image (image#1)
    ///   \param: num_descriptors1 - number of descriptors from the reference image
    ///   \param: dct_coeff_num - number of DCT coefficient used for CSFT descrpitor
    ///   \return: VNCC scores
    ///
    ///-- Copy the expanded matching descriptors to GPU memory
    ///
    size_t size_all_descs2_expanded = all_descs2_expanded.rows * all_descs2_expanded.cols * sizeof(float);
    float *ptr_float_all_descs2_expanded = (float *)all_descs2_expanded.data;
    Npp32f *gpu_all_descs2_expanded;
    cudaMalloc((void**)&gpu_all_descs2_expanded, size_all_descs2_expanded);
    cudaMemcpy(gpu_all_descs2_expanded, ptr_float_all_descs2_expanded, size_all_descs2_expanded, cudaMemcpyHostToDevice);
    NppiSize nppi_size_all_descs2_expanded;
    nppi_size_all_descs2_expanded.width = all_descs2_expanded.cols;
    nppi_size_all_descs2_expanded.height = all_descs2_expanded.rows;

    ///-- Copy subvector template for each reference descriptor to GPU memory
    ///
    int ref_subvector_rows = all_ref_subvector.rows / num_descriptors1;
    size_t size_all_ref_subvector = all_ref_subvector.rows * all_ref_subvector.cols * sizeof(float);
    float *ptr_float_all_ref_subvector = (float *)all_ref_subvector.data;
    Npp32f *gpu_all_ref_subvector;
    cudaMalloc((void**)&gpu_all_ref_subvector, size_all_ref_subvector);
    cudaMemcpy(gpu_all_ref_subvector, ptr_float_all_ref_subvector, size_all_ref_subvector, cudaMemcpyHostToDevice);
    NppiSize nppi_size_ref_subvector;  // nppi size for one reference subvector template
    nppi_size_ref_subvector.width = all_ref_subvector.cols;
    nppi_size_ref_subvector.height = ref_subvector_rows;

    ///-- Compute distance between each descriptor in reference image and that in matching image using VNCC
    ///
    Npp32f *gpu_all_corr_result;
    cudaMalloc((void**)&gpu_all_corr_result, num_descriptors1*(all_descs2_expanded.rows-(ref_subvector_rows-1))*sizeof(Npp32f));
    float *all_corr_result_data = static_cast<float*>(malloc(num_descriptors1*(all_descs2_expanded.rows-(ref_subvector_rows-1))*sizeof(float)));
    for (int i = 0; i < num_descriptors1; i++) {       
        int mat_line_step = (dct_coeff_num - 1) * sizeof(Npp32f);  // number of bytes between successive rows in the image
        nppiCrossCorrValid_Norm_32f_C1R(gpu_all_descs2_expanded, mat_line_step, nppi_size_all_descs2_expanded, gpu_all_ref_subvector+(i*ref_subvector_rows*(dct_coeff_num-1)), mat_line_step, nppi_size_ref_subvector, gpu_all_corr_result+(i*(all_descs2_expanded.rows-(ref_subvector_rows-1))), sizeof(Npp32f)); 
    }
    cudaMemcpy(all_corr_result_data, gpu_all_corr_result, num_descriptors1*(all_descs2_expanded.rows-(ref_subvector_rows-1))*sizeof(Npp32f), cudaMemcpyDeviceToHost);
    cudaFree(gpu_all_ref_subvector);   // free CUDA memory
    cudaFree(gpu_all_descs2_expanded); // free CUDA memory
    cudaFree(gpu_all_corr_result);     // free CUDA memory

    return all_corr_result_data;
}