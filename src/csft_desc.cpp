/**************************************************************************
* Program Name: Compressed Spatial Frequency Transfrom (CSFT)
*
* Filename: csft_desc.cpp
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

#include <csft_desc.h>

#if USE_CUDA
#include "ncc_gpu.h"
#else
#include "ncc_cpu.h"
#endif

#ifdef _OPENMP
#define OMP_MAX_THREADS 16
#endif


CSFTDescriptor::CSFTDescriptor()
{
    //-- Set initial value for member variables
    min_crop_size_ = 16;                                         // minimum crop size around a feature keypoint; (>= 16; even size); default: 16
    max_crop_size_ = 100;                                        // maximum crop size around a feature keypoint; (>= 16; even size); default: 100
    patch_size_ = static_cast<int>(ceil(max_crop_size_ * 1.42)); // size of original local patch around feature point used to perform rotation and crop
    rot_step_ = 15;                                              // step size (degrees) for rotating a crop; default: 30
    num_rots_ = static_cast<int>(ceil(90 / rot_step_));          // number of rotations for each crop encoded in the original (compact) descriptor
    crop_scale_factor_ = 1.2;                                    // scale factor between crop sizes; default: 1.5
    dct_coeff_num_ = 21;                                         // number of DCT coefficient (each crop) to use to create descrpitor; default: 21
    match_ratio_thresh_ = 0.7;                                   // ratio between the best match and the second best match; default: 0.7
    good_match_percent_ = 0.98;                                  // percentage of good feature matches based on spatial locality; default: 0.98
    //-- Determine the number of center crops
    num_crops_ = 0;
    int crop_size = min_crop_size_;
    while (crop_size < max_crop_size_) {
        crop_sizes_.push_back(crop_size);
        num_crops_++;
        crop_size = static_cast<int>(ceil(crop_size * crop_scale_factor_));
        (crop_size % 2 == 1) ? (crop_size += 1) : (crop_size); // make sure crop_size is an even number
    }
}


CSFTDescriptor::~CSFTDescriptor() {}


cv::Mat CSFTDescriptor::ImRotate(cv::Mat &img, float &angle, float &scale)
{
    ///
    ///-- Rotate image in spatial domain
    ///   \param: img - image to rotate
    ///   \param: angle - rotation angle in degrees
    ///   \param: scale - isotropic scale factor
    ///   \return: rotated image
    ///
    cv::Mat img_r = cv::Mat::zeros(img.size(), img.type());
    cv::Point2f pt_center(round(img.cols / 2), round(img.rows / 2));
    cv::Mat affine_mat = cv::getRotationMatrix2D(pt_center, angle, scale);  // calculate an affine matrix of 2D rotation.
    cv::warpAffine(img, img_r, affine_mat, img_r.size());
    return img_r;
}


void CSFTDescriptor::IndexZigzagScan(int &rows, int &cols)
{
    ///
    ///-- Compute the index of each DCT coefficent in zigzag order
    ///   \param: rows - number of rows in the DCT matrix
    ///   \param: cols - number of cols in the DCT matrix
    ///
    int row = 0, col = 0, cnt = 0;
    while (row < rows && col < cols && cnt < dct_coeff_num_) {
        if (row == 0 && (col % 2) == 0 && col != cols - 1) {
            //-- move right in the first row
            dct_coeff_loc_to_idx_.insert({{row, col}, cnt});
            dct_coeff_idx_to_loc_.insert({cnt, {row, col}});
            col++;
            cnt++;
        }
        else if (row == (rows - 1) && ((row + col) % 2) != 0 && col != cols - 1) {
            //-- move right in the last row
            dct_coeff_loc_to_idx_.insert({{row, col}, cnt});
            dct_coeff_idx_to_loc_.insert({cnt, {row, col}});
            col++;
            cnt++;
        }
        else if (col == 0 && ((row + col) % 2) != 0 && row != rows - 1) {
            //-- move down in the first column
            dct_coeff_loc_to_idx_.insert({{row, col}, cnt});
            dct_coeff_idx_to_loc_.insert({cnt, {row, col}});
            row++;
            cnt++;
        }
        else if (col == cols - 1 && ((row + col) % 2) == 0 && row != rows - 1) {
            //-- move down in the last column
            dct_coeff_loc_to_idx_.insert({{row, col}, cnt});
            dct_coeff_idx_to_loc_.insert({cnt, {row, col}});
            row++;
            cnt++;
        }
        else if (row != rows - 1 && col != 0 && ((row + col) % 2) != 0) {
            //-- move lower left along the diagonal
            dct_coeff_loc_to_idx_.insert({{row, col}, cnt});
            dct_coeff_idx_to_loc_.insert({cnt, {row, col}});
            row++;
            col--;
            cnt++;
        }
        else if (row != 0 && col != cols - 1 && ((row + col) % 2) == 0) {
            //-- move upper right along the diagonal
            dct_coeff_loc_to_idx_.insert({{row, col}, cnt});
            dct_coeff_idx_to_loc_.insert({cnt, {row, col}});
            row--;
            col++;
            cnt++;
        }
        else if (row == rows - 1 && col == cols - 1) {
            //-- the coefficient at the lower right corner
            dct_coeff_loc_to_idx_.insert({{row, col}, cnt});
            dct_coeff_idx_to_loc_.insert({cnt, {row, col}});
            break;
        }
    }
}


std::vector<float> CSFTDescriptor::ZigzagScan(cv::Mat &img_dct)
{
    ///
    ///-- Convert a DCT matrix into a 1-D array in zigzag order
    ///   \param: img_dct - DCT matrix
    ///   \return: 1-D vector containing the DCT coefficients in zigzag order
    ///
    std::vector<float> dct_coeffs(dct_coeff_num_, 0);
    for (auto i : dct_coeff_loc_to_idx_) {
        auto dct_coeff_loc = i.first;
        dct_coeffs[i.second] = img_dct.at<float>(dct_coeff_loc.first, dct_coeff_loc.second);
    }
    return dct_coeffs;
}


cv::Mat CSFTDescriptor::NormMat(cv::Mat &mat_in)
{
    ///
    ///-- Normalize the input matrix by subtracting the mean of the matrix intensities and dividing by the stddev
    ///   \param: mat_in - input matrix
    ///   \return: normalized matrix
    ///
    cv::Mat mat_norm(mat_in.size(), CV_32FC1);
    cv::Scalar mean_s, stdev_s;
    cv::meanStdDev(mat_in, mean_s, stdev_s);
    float mean_f = mean_s[0];
    float stdev_f = stdev_s[0];
    cv::Mat temp_mat1(mat_in.size(), CV_32FC1);
    cv::Mat temp_mat2(mat_in.size(), CV_32FC1);
    temp_mat1.setTo(mean_f);
    temp_mat2.setTo(stdev_f);
    cv::subtract(mat_in, temp_mat1, mat_norm);
    cv::divide(mat_norm, temp_mat2, mat_norm);
    return mat_norm;
}


cv::Mat CSFTDescriptor::Rotate90DCT(const cv::Mat &dct_coeffs)
{
    ///
    ///-- Compute DCT coefficients for an image rotated by 90 degrees using DCT geometric transformations
    ///   \param: dct_coeffs - DCT coefficients for the image before rotation
    ///   \return: DCT coefficients for the image after rotation
    ///
    cv::Mat dct_coeffs_transformed(dct_coeffs.size(), CV_32FC1);
    for (int i = 0; i < dct_coeffs.cols; ++i) {
        float dct_coeff_curr = dct_coeffs.at<float>(0, i);
        float dct_coeff_transformed = dct_coeff_curr;
        int row_idx = dct_coeff_idx_to_loc_[i + 1].first; // i+1 rather than i
        int col_idx = dct_coeff_idx_to_loc_[i + 1].second;
        if (col_idx % 2 == 1) {
            dct_coeff_transformed *= -1.0;
        }
        std::pair<int, int> dct_coeff_loc_transformed(col_idx, row_idx); // flip rows/cols
        int dct_coeff_idx_transformed = dct_coeff_loc_to_idx_[dct_coeff_loc_transformed] - 1;
        dct_coeffs_transformed.at<float>(0, dct_coeff_idx_transformed) = dct_coeff_transformed;
    }
    return dct_coeffs_transformed;
}


cv::Mat CSFTDescriptor::Rotate180DCT(const cv::Mat &dct_coeffs)
{
    ///
    ///-- Compute DCT coefficients for an image rotated by 180 degrees using DCT geometric transformations
    ///   \param: dct_coeffs - DCT coefficients for the image before rotation
    ///   \return: DCT coefficients for the image after rotation
    ///
    cv::Mat dct_coeffs_transformed(dct_coeffs.size(), CV_32FC1);
    for (int i = 0; i < dct_coeffs.cols; ++i) {
        float dct_coeff_curr = dct_coeffs.at<float>(0, i);
        int row_idx = dct_coeff_idx_to_loc_[i + 1].first; // i+1 rather than i
        int col_idx = dct_coeff_idx_to_loc_[i + 1].second;
        float dct_coeff_transformed = dct_coeff_curr * cos(PI * row_idx) * cos(PI * col_idx);
        dct_coeffs_transformed.at<float>(0, i) = dct_coeff_transformed;
    }
    return dct_coeffs_transformed;
}


cv::Mat CSFTDescriptor::Rotate270DCT(const cv::Mat &dct_coeffs)
{
    ///
    ///-- Compute DCT coefficients for an image rotated by 270 degrees using DCT geometric transformations
    ///   \param: dct_coeffs - DCT coefficients for the image before rotation
    ///   \return: DCT coefficients for the image after rotation
    ///
    cv::Mat dct_coeffs_transformed(dct_coeffs.size(), CV_32FC1);
    for (int i = 0; i < dct_coeffs.cols; ++i) {
        float dct_coeff_curr = dct_coeffs.at<float>(0, i);
        float dct_coeff_transformed = dct_coeff_curr;
        int row_idx = dct_coeff_idx_to_loc_[i + 1].first; // i+1 rather than i
        int col_idx = dct_coeff_idx_to_loc_[i + 1].second;
        if (row_idx % 2 == 1) {
            dct_coeff_transformed *= -1.0;
        }
        std::pair<int, int> dct_coeff_loc_transformed(col_idx, row_idx); // flip rows/cols
        int dct_coeff_idx_transformed = dct_coeff_loc_to_idx_[dct_coeff_loc_transformed] - 1;
        dct_coeffs_transformed.at<float>(0, dct_coeff_idx_transformed) = dct_coeff_transformed;
    }
    return dct_coeffs_transformed;
}


cv::Mat CSFTDescriptor::DescriptorExpand(cv::Mat &desc_orig)
{
    ///
    ///-- Expand the CSFT descriptor to full size
    ///   \param: desc_orig - compact CSFT descriptor
    ///   \return: expanded CSFT descriptor
    ///
    cv::Mat desc_expanded = cv::Mat::zeros(desc_orig.rows * 4, desc_orig.cols, CV_32FC1); // expanded descriptor will be 4 times the size of the original
    for (int i = 0; i < desc_orig.rows; ++i) {
        //-- First copy DCT coefficients to the same location in desc_expanded without doing any transformation
        cv::Mat desc_orig_curr_row(desc_orig(cv::Rect(0, i, desc_orig.cols, 1)));
        cv::Mat desc_expanded_dst_row_0(desc_expanded(cv::Rect(0, i, desc_orig.cols, 1)));
        desc_orig_curr_row.copyTo(desc_expanded_dst_row_0);
        //-- Compute the DCT coefficients for the corresponding image crop rotated by 90 degrees and copy to desc_expanded
        cv::Mat desc_orig_curr_row_transformed = Rotate90DCT(desc_orig_curr_row);
        cv::Mat desc_expanded_dst_row_90(desc_expanded(cv::Rect(0, desc_orig.rows + i, desc_orig.cols, 1))); // destination location in desc_expanded
        desc_orig_curr_row_transformed.copyTo(desc_expanded_dst_row_90);
        //-- Compute the DCT coefficients for the corresponding image crop rotated by 180 degrees and copy to desc_expanded
        desc_orig_curr_row_transformed = Rotate180DCT(desc_orig_curr_row);
        cv::Mat desc_expanded_dst_row_180(desc_expanded(cv::Rect(0, 2 * desc_orig.rows + i, desc_orig.cols, 1))); // destination location in desc_expanded
        desc_orig_curr_row_transformed.copyTo(desc_expanded_dst_row_180);
        //-- Compute the DCT coefficients for the corresponding image crop rotated by 270 degrees and copy to desc_expanded
        desc_orig_curr_row_transformed = Rotate270DCT(desc_orig_curr_row);
        cv::Mat desc_expanded_dst_row_270(desc_expanded(cv::Rect(0, 3 * desc_orig.rows + i, desc_orig.cols, 1))); // destination location in desc_expanded
        desc_orig_curr_row_transformed.copyTo(desc_expanded_dst_row_270);
    }
    return desc_expanded;
}


std::vector<cv::DMatch> CSFTDescriptor::CleanMatches(std::vector<cv::DMatch> matches_mto1)
{
    ///
    ///-- Find multiple-to-one matches, keep the strongest match and remove the others, so only one-to-one matches exist 
    ///   \param: matches_mto1 - vector of feature matches that may contain multiple-to-one matches
    ///   \return: one-to-one matches
    ///
    std::vector<cv::DMatch> matches;
    std::vector<bool> list_matches_to_keep(matches_mto1.size(), true);
    for (size_t i = 0; i < matches_mto1.size(); i++) {
        if (!list_matches_to_keep[i]) {
            continue;
        }
        int train_idx = matches_mto1[i].trainIdx; // current train index
        float dist_min = matches_mto1[i].distance;
        size_t dist_min_idx = i;
        for (size_t j = i + 1; j < matches_mto1.size(); j++) {
            if (matches_mto1[j].trainIdx == train_idx) { // multiple-to-one match found
                if (matches_mto1[j].distance < dist_min) { // only keep the match with minimum matching distance
                    dist_min = matches_mto1[j].distance;
                    list_matches_to_keep[dist_min_idx] = false;
                    dist_min_idx = j;
                }
                else {
                    list_matches_to_keep[j] = false;
                }
            }
        }
    }
    for (size_t i = 0; i < list_matches_to_keep.size(); i++) {
        if (list_matches_to_keep[i]) {
            matches.push_back(matches_mto1[i]);
        }
    }
    std::vector<bool>().swap(list_matches_to_keep); // release memory
    return matches;
}


std::vector<cv::DMatch> CSFTDescriptor::RefineMatches(std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> matches)
{
    ///
    ///-- Refine feature matches using spatial locality constraint
    ///   \param: keypoints1 - feature keypoints in image#1
    ///   \param: keypoints2 - feature keypoints in image#2
    ///   \param: matches - feature matches (one-to-one matches only)
    ///   \return: refined feature matches in OpenCV cv::DMatch data structure
    ///
    std::vector<cv::DMatch> matches_good;
    //-- create pair for each match and its spatial distance of the matched keypoints
    std::vector<std::pair<cv::DMatch, float> > vec_pair_match_dist;
    for (size_t i = 0; i < matches.size(); ++i) {
        cv::KeyPoint keypt_query = keypoints1[matches[i].queryIdx];
        cv::KeyPoint keypt_train = keypoints2[matches[i].trainIdx];
        float keypts_dist = std::sqrt(std::pow((keypt_query.pt.x-keypt_train.pt.x), 2.0) + std::pow((keypt_query.pt.y-keypt_train.pt.y), 2.0));
        std::pair<cv::DMatch, float> pair_match_dist (matches[i], keypts_dist);
        vec_pair_match_dist.push_back(pair_match_dist);
    }
    //-- sort matches based on the spatial distance of the matched keypoints
    std::sort(vec_pair_match_dist.begin(), vec_pair_match_dist.end(), SortMatchBySpatialDist());
    //-- delete the matches whose keypoints spatial distance fall into n percentage at the bottom
    int num_matches_to_keep = static_cast<int>(matches.size()*good_match_percent_);
    vec_pair_match_dist.erase(vec_pair_match_dist.begin()+num_matches_to_keep, vec_pair_match_dist.end());
    for (size_t i = 0; i < vec_pair_match_dist.size(); ++i) {
        matches_good.push_back(vec_pair_match_dist[i].first);
    }
    // free up memory
    std::vector<std::pair<cv::DMatch, float> >().swap(vec_pair_match_dist);
    return matches_good;
}


cv::Mat CSFTDescriptor::DescriptorExtract(cv::Mat &img_gray, std::vector<cv::KeyPoint> &keypoints, std::string csft_desc_type)
{
    ///
    ///-- Extract compact CSFT descriptors for keypoints
    ///   \param: img_gray - input grayscale image
    ///   \param: keypoints - detected keypoints
    ///   \param: csft_desc_type - CSFT descriptor type (Upright or Standard). Upright-CSFT disables rotation invariance
    ///   \return: CSFT descriptors
    ///
    if (csft_desc_type == "Upright") { num_rots_ = 1; }
    IndexZigzagScan(min_crop_size_, min_crop_size_); // create index for each DCT coefficient (x,y) in DCT matrix using zigzag scan
    cv::Mat img_gray32f;
    img_gray.convertTo(img_gray32f, CV_32FC1); // convert input grayscale image to floating point
    //-- add border around the grayscale image (replicate border type)
    int patch_radius = patch_size_ / 2;
    cv::Mat img_gray32f_pad = cv::Mat::zeros(img_gray.size(), CV_32FC1);  // size will change afterwards
    cv::copyMakeBorder(img_gray32f, img_gray32f_pad, patch_radius, patch_radius, patch_radius, patch_radius, cv::BORDER_REPLICATE);
    //-- create a descriptor for each feature keypoint
    cv::Mat descriptors = cv::Mat::zeros(keypoints.size(), (dct_coeff_num_ - 1) * num_rots_ * num_crops_, CV_32FC1); // initialize descriptor matrix
    int i;
#ifdef _OPENMP
    omp_set_num_threads(OMP_MAX_THREADS);
#pragma omp parallel for
#endif
    for (i = 0; i < keypoints.size(); i++) {
        int cnt = 0;
        int pt_x = keypoints[i].pt.x, pt_y = keypoints[i].pt.y;
        cv::Mat patch;
        //-- if the feature point lies within the boundary area, crop a local patch from the padded image
        if ((pt_x - patch_radius < 0) || (pt_x + patch_radius >= img_gray.cols) || (pt_y - patch_radius < 0) || (pt_y + patch_radius >= img_gray.rows)) {
            pt_x = pt_x + patch_radius;
            pt_y = pt_y + patch_radius;
            cv::Mat patch_temp(img_gray32f_pad(cv::Rect(pt_x - patch_radius, pt_y - patch_radius, patch_size_, patch_size_))); // crop a patch around the feature point
            patch = patch_temp.clone();
            patch_temp.release();
        }
        else {
            cv::Mat patch_temp(img_gray32f(cv::Rect(pt_x - patch_radius, pt_y - patch_radius, patch_size_, patch_size_))); // crop a patch around the feature point
            patch = patch_temp.clone();
            patch_temp.release();
        }
        //-- rotate the patch by a set of angles
        for (int j = 0; j < num_rots_; ++j) {
            cv::Mat patch_rotated;
            float scale = 1.0;
            float angle = rot_step_ * j;
            if (j == 0) {
                patch_rotated = patch.clone(); // do not rotate if the angle is 0
            }
            else {
                patch_rotated = ImRotate(patch, angle, scale); // rotate but keep the scale unchanged
            }
            //-- center crops from each rotated patch
            for (size_t k = 0; k < crop_sizes_.size(); ++k, cnt++) {
                int crop_size = crop_sizes_[k];
                cv::Mat crop(patch_rotated(cv::Rect(round(patch_rotated.cols / 2) - (crop_size / 2), round(patch_rotated.rows / 2) - (crop_size / 2), crop_size, crop_size)));
                cv::Mat crop_dct;
                cv::dct(crop, crop_dct);  // apply DCT
                std::vector<float> dct_coeffs = ZigzagScan(crop_dct); // obtain the first dct_coeff_num_ DCT coefficients in DCT matrix in zigzag order
                for (int c = 1; c < dct_coeff_num_; c++) {
                    //-- create a feature descriptor
                    descriptors.at<float>(i, (dct_coeff_num_ - 1) * cnt + c - 1) = dct_coeffs[c] / dct_coeffs[0]; // select and normalize the first (dct_coeff_num_-1) AC coefficients
                }
            }
        }
    }
    return descriptors;
}


std::vector<cv::DMatch> CSFTDescriptor::KNNMatcher(cv::Mat &descriptors1, cv::Mat &descriptors2)
{
    ///
    ///-- Compute feature matches using standard distance ratio matching (for Upright-CSFT)
    ///   \param: descriptors1 - descriptors for keypoints in the 1st image (reference image)
    ///   \param: descriptors2 - descriptors for keypoints in the 2nd image (matching image)
    ///   \return: feature matches
    ///
    bool flag_clean_matches = true;  // flag for removing multiple-to-one matches and keeping only one-to-one matches
    std::vector<cv::DMatch> matches, matches_mto1;
    std::vector<std::vector<cv::DMatch>> matches_all;
    cv::BFMatcher bf_matcher(cv::NORM_L2);
    bf_matcher.knnMatch(descriptors1, descriptors2, matches_all, 2); // find two nearest matches
    for (size_t i = 0; i < matches_all.size(); i++) {
        if (matches_all[i][0].distance < match_ratio_thresh_ * matches_all[i][1].distance) {
            matches_mto1.push_back(matches_all[i][0]);
        }
    }
    std::vector<std::vector<cv::DMatch>>().swap(matches_all);
    ///-- Find multiple-to-one matches, keep the strongest match and remove the others
    ///-- Make sure only one-to-one matches exist in matches
    ///
    if (flag_clean_matches) {
        matches = CleanMatches(matches_mto1);
    }
    else {
        matches = matches_mto1;
    }
    std::vector<cv::DMatch>().swap(matches_mto1);  // release memory
    return matches;
}


std::vector<cv::DMatch> CSFTDescriptor::VNCCMatcher(std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2, cv::Mat &descriptors1, cv::Mat &descriptors2)
{
    ///
    ///-- Compute feature matches using VNCC + distance ratio matching (for standard CSFT)
    ///   \param: keypoints1 - keypoints in the 1st image (reference image)
    ///   \param: keypoints2 - keypoints in the 2nd image (matching image)
    ///   \param: descriptors1 - descriptors for keypoints in the 1st image (reference image)
    ///   \param: descriptors2 - descriptors for keypoints in the 2nd image (matching image)
    ///   \return: feature matches
    ///
    bool flag_clean_matches = true;  // flag for removing multiple-to-one matches and keeping only one-to-one matches
    bool flag_refine_matches = true;  // flag for refining feature matches using spatial locality constraint
    std::vector<cv::DMatch> matches_mto1, matches, matches_good;

    ///-- Expand and normalize descriptors for all features in the matching image (2nd image) 
    ///
    int num_rows_desc_expanded = 4*descriptors2.cols/(dct_coeff_num_-1);
    cv::Mat all_descs2_expanded = cv::Mat::zeros(descriptors2.rows*num_rows_desc_expanded, dct_coeff_num_-1, CV_32FC1);
    int i;
#ifdef _OPENMP
    omp_set_num_threads(OMP_MAX_THREADS);
#pragma omp parallel for
#endif
    for (i = 0; i < descriptors2.rows; ++i) {
        cv::Mat desc2_orig(descriptors2(cv::Rect(0, i, descriptors2.cols, 1)));
        desc2_orig = desc2_orig.reshape(1, desc2_orig.cols / (dct_coeff_num_ - 1)); // reshape(#channels, #rows)
        cv::Mat desc2_expanded = DescriptorExpand(desc2_orig);  // expand the compressed descriptor to full size
        cv::Mat all_descs2_expanded_roi(all_descs2_expanded(cv::Rect(0, i*desc2_expanded.rows, desc2_expanded.cols, desc2_expanded.rows)));
        desc2_expanded.copyTo(all_descs2_expanded_roi);
        desc2_expanded.release();  // release memory
    }

    ///-- Extract the subvector template for each descriptor in the reference image
    ///
    int ref_subvector_rows = 3; // fixed parameter value: 3
    cv::Mat all_ref_subvector = cv::Mat::zeros(descriptors1.rows*ref_subvector_rows, dct_coeff_num_-1, CV_32FC1);
#ifdef _OPENMP
    omp_set_num_threads(OMP_MAX_THREADS);
#pragma omp parallel for
#endif    
    for (i = 0; i < descriptors1.rows; i++) {
        cv::Mat desc1_orig(descriptors1(cv::Rect(0, i, descriptors1.cols, 1)));
        desc1_orig = desc1_orig.reshape(1, desc1_orig.cols / (dct_coeff_num_ - 1));  // reshape(#channels, #rows)
        int ref_subvector_mid_idx = static_cast<int>(floor(desc1_orig.rows / (num_rots_ * 2)));
        cv::Mat ref_subvector(desc1_orig(cv::Rect(0, ref_subvector_mid_idx-static_cast<int>(floor(ref_subvector_rows/2)), dct_coeff_num_-1, ref_subvector_rows)));
        cv::Mat all_ref_subvector_roi(all_ref_subvector(cv::Rect(0, i*ref_subvector_rows, ref_subvector.cols, ref_subvector.rows)));
        ref_subvector.copyTo(all_ref_subvector_roi);
        desc1_orig.release();            // release memory
        ref_subvector.release();         // release memory
        all_ref_subvector_roi.release(); // release memory
    }

    ///-- Compute distance between each descriptor in reference image and that in the matching image using VNCC
    ///
#if USE_CUDA
    float *all_corr_result_ptr = NccCuda(all_descs2_expanded, all_ref_subvector, descriptors1.rows, dct_coeff_num_);
    cv::Mat all_corr_result(descriptors1.rows, all_descs2_expanded.rows-(ref_subvector_rows-1), CV_32FC1, all_corr_result_ptr);  // copy to a cv::Mat
#else
    cv::Mat all_corr_result = NccCpu(all_descs2_expanded, all_ref_subvector, descriptors1.rows, dct_coeff_num_);
#endif

    ///-- Find feature matches from VCNN score matrix using distance ratio matching strategy
    ///
#ifdef _OPENMP
    omp_set_num_threads(OMP_MAX_THREADS);
#pragma omp parallel
#endif
    {   
        std::vector<cv::DMatch> matches_mto1_private;
#ifdef _OPENMP
#pragma omp for nowait
#endif         
        for (i = 0; i < descriptors1.rows; i++) {    
            cv::Mat corr_result(all_corr_result(cv::Rect(0, i, all_descs2_expanded.rows-(ref_subvector_rows-1), 1)));  // get the VNCC score for current reference descriptor
            cv::transpose(corr_result, corr_result);  // convert to n-by-1 matrix
            //-- find the invalid correlation scores and set them to zero
            for (int k = 1; k < all_descs2_expanded.rows / num_crops_; ++k) {
                corr_result.at<float>(k * num_crops_ - 2, 0) = 0;
                corr_result.at<float>(k * num_crops_ - 1, 0) = 0;
            }
            //-- zero-pad correlation score matrix to make it the same size as all_descs2_expanded
            cv::Mat corr_result_pad = cv::Mat::zeros(all_descs2_expanded.rows, 1, CV_32FC1);
            cv::Mat corr_result_pad_roi(corr_result_pad(cv::Rect(0, 1, 1, corr_result.rows)));
            corr_result.copyTo(corr_result_pad_roi);
            //-- find the nearest neighbor, i.e. the feature with the minimum matching distance
            double max_val;
            cv::Point max_loc;
            cv::minMaxLoc(corr_result_pad, NULL, &max_val, NULL, &max_loc);                   // maximum correlation score = minimum matching distance
            double match_dist_min = ((1 - max_val) == 0) ? 0.00001 : 1 - max_val;             // make sure the distance is non-zero to avoid NaN
            int match_dist_min_idx = static_cast<int>(floor(max_loc.y / num_rows_desc_expanded)); // nearest neighbor feature ID in the matching image
            //-- delete correlation scores associated with the nearest neighbor
            for (int j = match_dist_min_idx * num_rows_desc_expanded; j < (match_dist_min_idx + 1) * num_rows_desc_expanded; ++j) {
                corr_result_pad.at<float>(j, 0) = 0.0;
            }
            //-- find the 2nd nearest neighbor, i.e. the feature with the 2nd minimum matching distance
            double second_max_val;
            cv::minMaxLoc(corr_result_pad, NULL, &second_max_val, NULL, NULL);
            double match_dist_min_2nd = ((1 - second_max_val) == 0) ? 0.00001 : 1 - second_max_val; // make sure the distance is non-zero to avoid NaN
            //-- feature matching using distance ratio matching strategy
            if ((match_dist_min / match_dist_min_2nd) <= match_ratio_thresh_) {
                cv::DMatch match_curr;
                match_curr.queryIdx = i;
                match_curr.trainIdx = match_dist_min_idx;
                match_curr.distance = match_dist_min;
                matches_mto1_private.push_back(match_curr);  // fill matches_mto1_private in parallel
            }
            //-- release memory
            corr_result.release();
            corr_result_pad.release();
            corr_result_pad_roi.release();  
        }
#ifdef _OPENMP
#pragma omp critical
#endif              
        matches_mto1.insert(matches_mto1.end(), matches_mto1_private.begin(), matches_mto1_private.end());  // combines matches_mto1_private in the end (thread safe)
        std::vector<cv::DMatch>().swap(matches_mto1_private);  // release memory
    }
    //-- release memory
    all_descs2_expanded.release();
    all_ref_subvector.release();
    all_corr_result.release();  // cv::Mat all_corr_result does not owns its data, so data is not released
#if USE_CUDA
    delete [] all_corr_result_ptr;  // need to explicitly release data in cv::Mat all_corr_result
#endif

    ///-- Find multiple-to-one matches, keep the strongest match and remove the others
    ///-- Make sure only one-to-one matches exist in matches
    ///
    matches = (flag_clean_matches)? CleanMatches(matches_mto1) : matches_mto1;
    std::vector<cv::DMatch>().swap(matches_mto1);  // release memory

    ///-- Refine feature matches using spatial locality constraint
    ///
    matches_good = (flag_refine_matches)? RefineMatches(keypoints1, keypoints2, matches) : matches;
    std::vector<cv::DMatch>().swap(matches);  // release memory

    return matches_good;
}
