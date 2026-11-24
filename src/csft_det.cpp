//=============================================================================
//
// csft_det.cpp
//
// Implemented on top of AKAZE.cpp
// 
// AKAZE Features:
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


#include "csft_det.h"
#include <opencv2/highgui/highgui.hpp>

using namespace std;

struct SortByResponse {
    bool operator() (cv::KeyPoint const &a, cv::KeyPoint const &b) {
        return a.response > b.response;
    }
};

/* ************************************************************************* */
CSFTDetector::CSFTDetector(const DetectorOptions& options) : options_(options) {

    img_pad_size_ = 50;    
    ncycles_ = 0;
    reordering_ = true;

    Allocate_Memory_Evolution();
}

/* ************************************************************************* */
CSFTDetector::~CSFTDetector() {
    evolution_.clear();
}

/* ************************************************************************* */
void CSFTDetector::Allocate_Memory_Evolution() {

    float rfactor = 0.0;
    int level_height = 0, level_width = 0;

    // Allocate the dimension of the matrices for the evolution
    for (int i = 0; i <= options_.omax-1; i++) {
        rfactor = 1.0/pow(2.0f, i);
        level_height = (int)((options_.img_height+2*img_pad_size_)*rfactor);
        level_width = (int)((options_.img_width+2*img_pad_size_)*rfactor);

        // Smallest possible octave and allow one scale if the image is small
        if ((level_width < 80 || level_height < 40) && i != 0) {
            options_.omax = i;
            break;
        }

        for (int j = 0; j < options_.nsublevels; j++) {
            TEvolution step;
            cv::Size size(level_width, level_height);
            step.Lx.create(size, CV_32F);
            step.Ly.create(size, CV_32F);
            step.Lxx.create(size, CV_32F);
            step.Lxy.create(size, CV_32F);
            step.Lyy.create(size, CV_32F);
            step.Lt.create(size, CV_32F);
            step.Lresp.create(size, CV_32F);
            step.Lflow.create(size, CV_32F);
            step.Lstep.create(size, CV_32F);

            step.esigma = options_.soffset*pow(2.0f, (float)(j)/(float)(options_.nsublevels) + i);
            step.sigma_size = fRound(step.esigma);
            step.etime = 0.5*(step.esigma*step.esigma);
            step.octave = i;
            step.sublevel = j;
            evolution_.push_back(step);
        }
    }

    // Allocate memory for the number of cycles and time steps
    for (size_t i = 1; i < evolution_.size(); i++) {
        int naux = 0;
        vector<float> tau;
        float ttime = 0.0;
        ttime = evolution_[i].etime-evolution_[i-1].etime;
        naux = fed_tau_by_process_time(ttime, 1, 0.25, reordering_,tau);
        nsteps_.push_back(naux);
        tsteps_.push_back(tau);
        ncycles_++;
    }
}

/* ************************************************************************* */
int CSFTDetector::Create_Nonlinear_Scale_Space(const cv::Mat& img) {

    if (evolution_.size() == 0) {
        cerr << "Error generating the nonlinear scale space!!" << endl;
        cerr << "Firstly you need to call CSFTDetector::Allocate_Memory_Evolution()" << endl;
        return -1;
    }

    // Add padding to the original image to ensure keypoints in the boundary area can be detected
    cv::Mat img_pad = cv::Mat::zeros(img.size(), CV_32FC1);  // size will change afterwards
    cv::copyMakeBorder(img, img_pad, img_pad_size_, img_pad_size_, img_pad_size_, img_pad_size_, cv::BORDER_REPLICATE);

    // Copy the original image to the first level of the evolution
    img_pad.copyTo(evolution_[0].Lt);
    gaussian_2D_convolution(evolution_[0].Lt, evolution_[0].Lt, 0, 0, options_.soffset);
    evolution_[0].Lt.copyTo(evolution_[0].Lsmooth);

    // First compute the kcontrast factor
    options_.kcontrast = compute_k_percentile(img_pad, options_.kcontrast_percentile,
                                              1.0, options_.kcontrast_nbins, 0, 0);


    // Now generate the rest of evolution levels
    for (size_t i = 1; i < evolution_.size(); i++) {

        if (evolution_[i].octave > evolution_[i-1].octave) {
            halfsample_image(evolution_[i-1].Lt, evolution_[i].Lt);
            options_.kcontrast = options_.kcontrast*0.75;
        }
        else {
            evolution_[i-1].Lt.copyTo(evolution_[i].Lt);
        }

        gaussian_2D_convolution(evolution_[i].Lt, evolution_[i].Lsmooth, 0, 0, 1.0);

        // Compute the Gaussian derivatives Lx and Ly
        image_derivatives_scharr(evolution_[i].Lsmooth, evolution_[i].Lx, 1, 0);
        image_derivatives_scharr(evolution_[i].Lsmooth, evolution_[i].Ly, 0, 1);

        // Compute the conductivity equation
        switch (options_.diffusivity) {
        case PM_G1:
            pm_g1(evolution_[i].Lx, evolution_[i].Ly, evolution_[i].Lflow, options_.kcontrast);
            break;
        case PM_G2:
            pm_g2(evolution_[i].Lx, evolution_[i].Ly, evolution_[i].Lflow, options_.kcontrast);
            break;
        case WEICKERT:
            weickert_diffusivity(evolution_[i].Lx, evolution_[i].Ly, evolution_[i].Lflow, options_.kcontrast);
            break;
        case CHARBONNIER:
            charbonnier_diffusivity(evolution_[i].Lx, evolution_[i].Ly, evolution_[i].Lflow, options_.kcontrast);
            break;
        default:
            cerr << "Diffusivity: " << options_.diffusivity << " is not supported" << endl;
        }

        // Perform FED n inner steps
        for (int j = 0; j < nsteps_[i-1]; j++)
            nld_step_scalar(evolution_[i].Lt, evolution_[i].Lflow, evolution_[i].Lstep, tsteps_[i-1][j]);
    }

    return 0;
}

/* ************************************************************************* */
void CSFTDetector::Feature_Detection(std::vector<cv::KeyPoint>& kpts) {

    vector<cv::KeyPoint>().swap(kpts);
    Compute_Hessian_Response();
    Find_Scale_Space_Extrema(kpts);
    Do_Subpixel_Refinement(kpts);
    Keep_N_Keypoints(kpts);
}

/* ************************************************************************* */
void CSFTDetector::Compute_Multiscale_Derivatives() {

#ifdef _OPENMP
    omp_set_num_threads(OMP_MAX_THREADS);
#pragma omp parallel for
#endif

    for (int i = 0; i < (int) evolution_.size(); i++) {

        float ratio = pow(2.0f,(float)evolution_[i].octave);
        int sigma_size_ = fRound(evolution_[i].esigma*options_.derivative_factor/ratio);

        compute_scharr_derivatives(evolution_[i].Lsmooth, evolution_[i].Lx, 1, 0, sigma_size_);
        compute_scharr_derivatives(evolution_[i].Lsmooth, evolution_[i].Ly, 0, 1, sigma_size_);
        compute_scharr_derivatives(evolution_[i].Lx, evolution_[i].Lxx, 1, 0, sigma_size_);
        compute_scharr_derivatives(evolution_[i].Ly, evolution_[i].Lyy, 0, 1, sigma_size_);
        compute_scharr_derivatives(evolution_[i].Lx, evolution_[i].Lxy, 0, 1, sigma_size_);
    }
}

/* ************************************************************************* */
void CSFTDetector::Compute_Hessian_Response() {

    // Firstly compute the multiscale derivatives
    Compute_Multiscale_Derivatives();

    for (size_t i = 0; i < evolution_.size(); i++) {

        if (options_.verbosity == true)
            cout << "Computing detector response. Evolution time: " << evolution_[i].etime << endl;

        float ratio = pow(2.0f,(float)evolution_[i].octave);
        int sigma_size = fRound(evolution_[i].esigma*options_.derivative_factor/ratio);
        int sigma_size_quat = sigma_size*sigma_size*sigma_size*sigma_size;

        for (int ix = 0; ix < evolution_[i].Lresp.rows; ix++) {
            const float* lxx = evolution_[i].Lxx.ptr<float>(ix);
            const float* lxy = evolution_[i].Lxy.ptr<float>(ix);
            const float* lyy = evolution_[i].Lyy.ptr<float>(ix);
            float* lresp = evolution_[i].Lresp.ptr<float>(ix);
            for (int jx = 0; jx < evolution_[i].Lresp.cols; jx++)
                lresp[jx] = pow((lxx[jx]+lyy[jx]),2.0f) / (fabs(lxx[jx]*lyy[jx]-lxy[jx]*lxy[jx])+1)*sigma_size_quat; // Trace*Trace/(abs(Det)+1)
        }
    }
}

/* ************************************************************************* */
void CSFTDetector::Find_Scale_Space_Extrema(std::vector<cv::KeyPoint>& kpts) {
    float value = 0.0;
    float dist = 0.0, ratio = 0.0, smax = 0.0;
    int npoints = 0, id_repeated = 0;
    int sigma_size_ = 0, left_x = 0, right_x = 0, up_y = 0, down_y = 0;
    int nms_radius = 6; // non-maximum suppresion region size on each side
    bool is_extremum = false, is_repeated = false, is_out = false;
    cv::KeyPoint point;
    vector<cv::KeyPoint> kpts_aux;

    smax = 10.0*sqrtf(2.0f);

    for (size_t i = 0; i < evolution_.size(); i++) {
        for (int ix = nms_radius; ix < evolution_[i].Lresp.rows-nms_radius; ix++) {
            float* lresp = evolution_[i].Lresp.ptr<float>(ix);
            for (int jx = nms_radius; jx < evolution_[i].Lresp.cols-nms_radius; jx++) {

                is_extremum = false;
                is_repeated = false;
                is_out = false;
                value = lresp[jx];
                bool is_localmax = true;

                // Filter the points with the detector threshold
                for (int i1 = ix-nms_radius; i1 <= ix+nms_radius; i1++) {
                    for (int j1 = jx-nms_radius; j1 <= jx+nms_radius; j1++) {
                        if (evolution_[i].Lresp.at<float>(i1,j1) > value) {
                            is_localmax = false;
                            goto here; // break a nested loop
                        }
                    }
                }
here:

                if (value > options_.dthreshold && value >= options_.min_dthreshold && is_localmax) {

                    is_extremum = true;
                    point.response = fabs(value);
                    point.size = evolution_[i].esigma*options_.derivative_factor;
                    point.octave = evolution_[i].octave;
                    point.class_id = i;
                    ratio = pow(2.0f, point.octave);
                    sigma_size_ = fRound(point.size/ratio);
                    point.pt.x = jx;
                    point.pt.y = ix;

                    // Compare response with the same and lower scale
                    for (size_t ik = 0; ik < kpts_aux.size(); ik++) {

                        if ((point.class_id-1) == kpts_aux[ik].class_id ||
                                point.class_id == kpts_aux[ik].class_id) {

                            dist = (point.pt.x*ratio-kpts_aux[ik].pt.x)*(point.pt.x*ratio-kpts_aux[ik].pt.x) +
                                    (point.pt.y*ratio-kpts_aux[ik].pt.y)*(point.pt.y*ratio-kpts_aux[ik].pt.y);

                            // if (dist <= point.size*point.size) {
                            if (dist <= (float)nms_radius*(float)nms_radius) {
                                if (point.response > kpts_aux[ik].response) {
                                    id_repeated = ik;
                                    is_repeated = true;
                                }
                                else {
                                    is_extremum = false;
                                }
                                break;
                            }
                        }
                    }

                    // Check out of bounds
                    if (is_extremum == true) {

                        // Check that the point is under the image limits for the descriptor computation
                        left_x = fRound(point.pt.x-smax*sigma_size_)-1;
                        right_x = fRound(point.pt.x+smax*sigma_size_) +1;
                        up_y = fRound(point.pt.y-smax*sigma_size_)-1;
                        down_y = fRound(point.pt.y+smax*sigma_size_)+1;

                        if (left_x < 0 || right_x >= evolution_[i].Lresp.cols ||
                                up_y < 0 || down_y >= evolution_[i].Lresp.rows) {
                            is_out = true;
                        }

                        if (is_out == false) {
                            if (is_repeated == false) {
                                point.pt.x = point.pt.x*ratio + .5*(ratio-1.0);
                                point.pt.y = point.pt.y*ratio + .5*(ratio-1.0);
                                kpts_aux.push_back(point);
                                npoints++;
                            }
                            else {
                                point.pt.x = point.pt.x*ratio + .5*(ratio-1.0);
                                point.pt.y = point.pt.y*ratio + .5*(ratio-1.0);
                                kpts_aux[id_repeated] = point;
                            }
                        } // if is_out
                    } //if is_extremum
                }
            } // for jx
        } // for ix
    } // for i

    // Now filter points with the all other scale levels
    for (size_t i = 0; i < kpts_aux.size(); i++) {

        is_repeated = false;
        const cv::KeyPoint& point = kpts_aux[i];
        for (size_t j = 0; j < kpts_aux.size(); j++) {
            if (j == i) {
                continue;
            }
            // Compare response with the all other scales
            dist = (point.pt.x-kpts_aux[j].pt.x)*(point.pt.x-kpts_aux[j].pt.x) +
                    (point.pt.y-kpts_aux[j].pt.y)*(point.pt.y-kpts_aux[j].pt.y);

            if (dist <= (float)nms_radius*(float)nms_radius) {
                if (point.response < kpts_aux[j].response) {
                    is_repeated = true;
                    break;
                }
            }
        }

        if (is_repeated == false)
            kpts.push_back(point);
    }
}

/* ************************************************************************* */
void CSFTDetector::Do_Subpixel_Refinement(std::vector<cv::KeyPoint>& kpts) {

    float Dx = 0.0, Dy = 0.0, ratio = 0.0;
    float Dxx = 0.0, Dyy = 0.0, Dxy = 0.0;
    int x = 0, y = 0;
    cv::Matx22f A(0.0, 0.0, 0.0, 0.0);
    cv::Vec2f b(0.0, 0.0);
    cv::Vec2f dst(0.0, 0.0);

    for (size_t i = 0; i < kpts.size(); i++) {
        ratio = pow(2.f,kpts[i].octave);
        x = fRound(kpts[i].pt.x/ratio);
        y = fRound(kpts[i].pt.y/ratio);

        // Compute the gradient
        Dx = (0.5)*(*(evolution_[kpts[i].class_id].Lresp.ptr<float>(y)+x+1)
                -*(evolution_[kpts[i].class_id].Lresp.ptr<float>(y)+x-1));
        Dy = (0.5)*(*(evolution_[kpts[i].class_id].Lresp.ptr<float>(y+1)+x)
                -*(evolution_[kpts[i].class_id].Lresp.ptr<float>(y-1)+x));

        // Compute the Hessian
        Dxx = (*(evolution_[kpts[i].class_id].Lresp.ptr<float>(y)+x+1)
                + *(evolution_[kpts[i].class_id].Lresp.ptr<float>(y)+x-1)
                -2.0*(*(evolution_[kpts[i].class_id].Lresp.ptr<float>(y)+x)));

        Dyy = (*(evolution_[kpts[i].class_id].Lresp.ptr<float>(y+1)+x)
                + *(evolution_[kpts[i].class_id].Lresp.ptr<float>(y-1)+x)
                -2.0*(*(evolution_[kpts[i].class_id].Lresp.ptr<float>(y)+x)));

        Dxy = (0.25)*(*(evolution_[kpts[i].class_id].Lresp.ptr<float>(y+1)+x+1)
                +(*(evolution_[kpts[i].class_id].Lresp.ptr<float>(y-1)+x-1)))
                -(0.25)*(*(evolution_[kpts[i].class_id].Lresp.ptr<float>(y-1)+x+1)
                +(*(evolution_[kpts[i].class_id].Lresp.ptr<float>(y+1)+x-1)));

        // Solve the linear system
        A(0,0) = Dxx;
        A(1,1) = Dyy;
        A(0,1) = A(1,0) = Dxy;
        b(0) = -Dx;
        b(1) = -Dy;

        cv::solve(A, b, dst, cv::DECOMP_LU);

        if (fabs(dst(0)) <= 1.0 && fabs(dst(1)) <= 1.0) {
            kpts[i].pt.x = x + dst(0);
            kpts[i].pt.y = y + dst(1);
            int power = powf(2, evolution_[kpts[i].class_id].octave);
            kpts[i].pt.x = kpts[i].pt.x*power + .5*(power-1);
            kpts[i].pt.y = kpts[i].pt.y*power + .5*(power-1);
            kpts[i].angle = 0.0;

            // In OpenCV the size of a keypoint its the diameter
            kpts[i].size *= 2.0;
        }
        // Delete the point since its not stable
        else {
            kpts.erase(kpts.begin()+i);
            i--;
        }
    }
}

/* ************************************************************************* */
void CSFTDetector::Keep_N_Keypoints(std::vector<cv::KeyPoint>& kpts) {
    //-- Remove the keypoints in the padding area and change the coordinates to the values without padding
    for (size_t i = 0; i < kpts.size(); ++i) {
        if (kpts[i].pt.x <= img_pad_size_ || kpts[i].pt.y <= img_pad_size_ || kpts[i].pt.x >= (options_.img_width+img_pad_size_) || kpts[i].pt.y >= (options_.img_height+img_pad_size_)) {
            kpts.erase(kpts.begin()+i);
            i--;
        }
        else {
            kpts[i].pt.x -= img_pad_size_; 
            kpts[i].pt.y -= img_pad_size_; 
        }
    }
    //-- Sort the keypoints based on response and keep the top N keypoints
    std::sort(kpts.begin(), kpts.end(), SortByResponse());
    if ((int)kpts.size() > options_.n_keypoints) {
        kpts.erase(kpts.begin()+options_.n_keypoints, kpts.end());
    }
}
