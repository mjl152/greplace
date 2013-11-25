#ifndef _ALPHA_FILTER_KERNEL_H
#define _ALPHA_FILTER_KERNEL_H

#include <opencv2/gpu/gpu.hpp>

extern "C"
void alphaKernelCaller(cv::gpu::PtrStepSz<uchar> & x, double r0, double rf);

extern "C" void reverseAlphaKernelCaller(cv::gpu::PtrStepSz<uchar> & x,
                                         double r0, double rf);

extern "C"
void diff(cv::gpu::PtrStepSz<uchar> & in, cv::gpu::PtrStepSz<uchar> & out);

#endif
