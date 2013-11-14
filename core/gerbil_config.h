#ifndef GERBIL_CONFIG_H
#define GERBIL_CONFIG_H

//***************************************************
//* Compile time generated configuration from CMake *
//***************************************************

// For now this file only contains self-contained pre-processor macros. We add
// gerbil_config.h.in later to get data from CMake variables.

// HAVE_OPENCV_GPU
// This pre-processor definition is passed as a compiler argument from CMake.
// See CMakeLists.txt in gerbil source root.

// HAVE_CUDA_GPU
// Pre-processor macro evaluating to true if a CUDA capable GPU supported by
// OpenCV is present.
// CMake < 2.4 does not have cv::gpu so we need to make sure the symbol is not
// used in the code.

#ifdef HAVE_OPENCV_GPU
#define HAVE_CUDA_GPU (cv::gpu::getCudaEnabledDeviceCount() > 0)
#else
#define HAVE_CUDA_GPU false
#endif


#endif // GERBIL_CONFIG_H
