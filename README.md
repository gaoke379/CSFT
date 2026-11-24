# Compressed Spatial Frequency Transform (CSFT)


**Copyright (C) 2018-2022. K. Gao, H. AliAkbarpour, K. Palaniappan and Curators of the University of Missouri, a public corporation.**   
**All Rights Reserved.**  

*For more information, contact:  
Ke Gao, gaoke379@gmail.com  
Prof. K. Palaniappan, palaniappank@missouri.edu  
226 Naka Hall  
University of Missouri-Columbia  
Columbia, MO 65211*  


## Patent

This repository contains code associated with the following issued U.S. patent:

**Gao, K., Akbarpour, H. A., and Palaniappan, K. “Compressed Spatial Frequency Transform for Feature Tracking, Image Matching, Search, and Retrieval.” U.S. Patent No. 12,755,509 B2, issued October 6, 2026.**

- **Assignee:** The Curators of the University of Missouri, Columbia, MO (US)
- **Application No.:** 18/300,182, filed April 13, 2023, claiming priority to Provisional Application No. 63/330,633, filed April 13, 2022
- **Publication No.:** US 2023/0334821 A1, published October 19, 2023
- **Claims:** 20

The patent covers a method and system for feature tracking and image matching using CSFT descriptors: detecting key points and associated patches in an image, transforming those patches from the spatial domain to the frequency domain using a Discrete Cosine Transform (DCT) or Discrete Fourier Transform (DFT), encoding illumination, rotation, scale, and other geometric transformations into a CSFT / DCTF descriptor, and matching a second image to the first image using those descriptors.

This invention was made with government support under W911NF-18-2-0285 and W911NF-19-1-0181, awarded by the Army Research Laboratory–Army Research Office. The government has certain rights in the invention.

Related publication: K. Gao, H. Ali Akbarpour, et al., “DCT-Based Local Descriptor for Robust Matching and Feature Tracking in Wide Area Motion Imagery,” *IEEE Geoscience and Remote Sensing Letters*, vol. 18, no. 8, pp. 1441–1445, Aug. 2021.


## Description
CSFT is novel feature detection and description approach for robustly matching features across different views suitable for applications such as camera pose estimation and Structure-from-Motion in large scale aerial images. CSFT features are robust across a range of scale and rotation, image blur, varying illumination, and JPEG compression artifacts. The CSFT descriptor is compressed using geometric transformations in the frequency domain that significantly reduce its size. The CSFT feature uses a novel vectorized Normalized Cross Correlation (VNCC) method for matching descriptors. CSFT is a domain independent method that can be easily implemented. CSFT requires no training step compared to the deep learning approaches.  



## Building CSFT

### Build Library Dependencies   
Before CSFT can be built, several dependencies need to be available on the system.     

- `OpenCV C++ 3.0.0 ~ 3.4.10`
- `CMake`
- `OpenMP`
- `CUDA (optional)`

`CUDA` is optional but recommended for a better run-time performance. CMake flag `-DNO_CUDA_BUILD` can be used to adjust the build to switch between GPU and CPU implementation as needed. CSFT is built without CUDA when set to `-DNO_CUDA_BUILD=1`. Otherwise, CMake defaults to building with CUDA support when not explicitly specified, i.e., `-DNO_CUDA_BUILD=0`.
 
### Configure and Build   
```
$ cd CSFT
$ mkdir build
$ cd build
$ cmake ..
$ make  
```

## Usage
### Running CSFT
Once CSFT has been successfully built, there should be a executable in the `bin` directory: `CSFT`.  

To run CSFT, go to `./bin/linux` for Linux or `./bin/macos` for MacOS, and invoke `./CSFT <parameters_config_file>`.  
Example usage: `./CSFT ../../params.txt`  

### Description of the CSFT Parameters in Config File
- **img1_fullpath**: path to the 1st input image
- **img2_fullpath**: path to the 2nd input image
- **out_path**: path to the output folder
- **n_keypoints**: number of keypoints to detect per image
- **csft_desc_type**: CSFT descriptor type (Upright or Standard). Upright-CSFT disables rotation invariance
- **flag_export_keypts**: flag for exporting feature keypoints to text files
- **flag_export_desc**: flag for exporting feature descriptors to text files
- **flag_export_match**: flag for exporting feature matches to text files
- **flag_draw_keypts_img**: flag for visualizing feature keypoints on the input images
- **flag_draw_match_img**: flag for visualizing feature matches on the input images  


## Output
The output files will be saved to the directory specified by `out_path` in the parameters config file.
### Visualization
- Images showing the detected keypoints will be saved if `flag_draw_keypts_img` is set to `1`.  
- An image showing the matches will be saved if `flag_draw_match_img` is set to `1`.  

### Text Files
- If `flag_export_keypts` is set to `1`, an ASCII text file containing the keypoints locations (`x y`) will be created for each input image where each row corresponds to a keypoint.   
Example:
```
618.116028 204.111755
437.704346 299.728333
597.402649 202.520157
188.808380 446.749207
437.238708 306.217590
```
- If `flag_export_desc` is set to `1`, an ASCII text file containing the descriptors will be created for each input image where each row corresponds to the descriptor for a keypoint.   
- If `flag_export_match` is set to `1`, an ASCII text file containing the matches (`x1 y1 x2 y2`) will be created for a pair of images where each row corresponds to a match. `x1 y1` and `x2 y2` denote the coordinate of the matched keypoint in the first image and second image, respectively.    
Example:
```
421.005707 304.459961 406.880188 321.907928
409.846680 297.340576 397.476959 325.259186
392.506470 306.396332 393.034668 338.806610
376.126678 328.696777 392.975586 357.134857
437.796295 402.756653 462.513184 370.383728
```


## Run-time Analysis  
Run-time in seconds for CSFT to perform keypoint detection, description, and matching (I/O included) for two images using the implementation with GPU support, multi-threaded CPU implementation with OpenMP, and single-threaded CPU implementation, respectively. 

### Platform 
Server with an 8-core Intel(R) Xeon(R) W-2123 CPU @ 3.60GHz and an NVIDIA Quadro RTX 8000 GPU.    

### Test data #1
- Reference image: `rit_rgb0000.jpg`
- Matching image: `rit_rgb0009.jpg`
- Image size: `1200 x 800`
- Number of keypoints per image: `3000`
- Number of matches: `1592`
- Run-time: 
  - GPU: `4.97s`
  - Multi-threaded CPU (OMP_MAX_THREADS=16): `40.01s`
  - Single-threaded CPU: `190.02s` 

### Test data #2
- Reference image: `boat_img1.png`
- Matching image: `boat_img3.png`
- Image size: `850 x 680`
- Number of keypoints per image: `200`
- Number of matches: `52`
- Run-time: 
  - GPU: `2.16s`
  - Multi-threaded CPU (OMP_MAX_THREADS=16): `0.70s`
  - Single-threaded CPU: `1.60s`

The implementation with GPU support achieves over `35x` speed-up compared to the single-threaded CPU implementation and over `8x` speed-up compared to the multi-threaded CPU implementation for test images with `3000` keypoints. However, the GPU implementation is slower than the CPU implementations for images with small amount of keypoints, since transferring memory between the host (CPU) and device (GPU) is time-consuming.  