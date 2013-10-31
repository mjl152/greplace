#ifndef _ALPHA_FILTER_KERNEL_H
#define _ALPHA_FILTER_KERNEL_H

#include <opencv2/gpu/gpu.hpp>

extern "C"
void alphaKernelCaller(cv::gpu::PtrStepSz<uchar> & x);

extern "C"
void diff(cv::gpu::PtrStepSz<uchar> & in, cv::gpu::PtrStepSz<uchar> & out);

#endif