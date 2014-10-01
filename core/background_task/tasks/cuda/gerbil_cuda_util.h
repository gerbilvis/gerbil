#ifndef GERBIL_CUDA_UTIL_H
#define GERBIL_CUDA_UTIL_H

#include <opencv2/core/gpumat.hpp>

/** Returns true if OpenCV reports there is a CUDA enabled GPU. */
inline bool haveCvCudaGpu() {
	return cv::gpu::getCudaEnabledDeviceCount() > 0;
}

#endif // GERBIL_CUDA_UTIL_H
