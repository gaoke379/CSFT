//=============================================================================
//
// csft_det.h
//
// Implemented on top of AKAZE.h from:
// Authors: Pablo F. Alcantarilla (1), Jesus Nuevo (2)
// Institutions: Toshiba Research Europe Ltd (1)
//               TrueVision Solutions (2)
// Date: 07/10/2014
// Email: pablofdezalc@gmail.com
//
// AKAZE Features Copyright 2014, Pablo F. Alcantarilla, Jesus Nuevo
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

#pragma once

/* ************************************************************************* */
#include "AKAZEConfig.h"
#include "fed.h"
#include "akaze_utils.h"
#include "nldiffusion_functions.h"

// OpenCV
#include <opencv2/features2d/features2d.hpp>

/* ************************************************************************* */
class CSFTDetector {

private:

    DetectorOptions options_;                      ///< Configuration options for CSFTDetector
    std::vector<TEvolution> evolution_;         ///< Vector of nonlinear diffusion evolution
    int img_pad_size_;                          // image padding size along each boundary

    /// FED parameters
    int ncycles_;                               ///< Number of cycles
    bool reordering_;                           ///< Flag for reordering time steps
    std::vector<std::vector<float > > tsteps_;  ///< Vector of FED dynamic time steps
    std::vector<int> nsteps_;                   ///< Vector of number of steps per cycle


public:

    /// CSFTDetector constructor with input options
    /// @param options CSFTDetector configuration options
    /// @note This constructor allocates memory for the nonlinear scale space
    CSFTDetector(const DetectorOptions& options);

    /// Destructor
    ~CSFTDetector();

    /// Allocate the memory for the nonlinear scale space
    void Allocate_Memory_Evolution();

    /// This method creates the nonlinear scale space for a given image
    /// @param img Input image for which the nonlinear scale space needs to be created
    /// @return 0 if the nonlinear scale space was created successfully, -1 otherwise
    int Create_Nonlinear_Scale_Space(const cv::Mat& img);

    /// @brief This method selects interesting keypoints through the nonlinear scale space
    /// @param kpts Vector of detected keypoints
    void Feature_Detection(std::vector<cv::KeyPoint>& kpts);

    /// This method computes the feature detector response for the nonlinear scale space
    /// @note We use the Hessian Trace*Trace/(abs(Det)+1) as the feature detector response
    void Compute_Hessian_Response();

    /// This method computes the multiscale derivatives for the nonlinear scale space
    void Compute_Multiscale_Derivatives();

    /// This method finds extrema in the nonlinear scale space
    void Find_Scale_Space_Extrema(std::vector<cv::KeyPoint>& kpts);

    /// This method performs subpixel refinement of the detected keypoints fitting a quadratic
    void Do_Subpixel_Refinement(std::vector<cv::KeyPoint>& kpts);

    /// This method sorts the keypoints wrt their responses and keep the first N keypoints
    void Keep_N_Keypoints(std::vector<cv::KeyPoint>& kpts);
};

/* ************************************************************************* */
/// This function computes the value of a 2D Gaussian function
inline float gaussian(float x, float y, float sigma) {
    return expf(-(x*x+y*y)/(2.0f*sigma*sigma));
}

/// This funtion rounds float to nearest integer
inline int fRound(float flt) {
    return (int)(flt+0.5f);
}
