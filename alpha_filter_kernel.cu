#ifndef _ALPHA_FILTER_KERNEL_CU
#define _ALPHA_FILTER_KERNEL_CU

#include <opencv2/gpu/gpu.hpp>

#include "stdio.h"

__global__ void alphaKernel(cv::gpu::PtrStepSz<uchar> mat, double max_dist) 
{ 
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < mat.cols && y < mat.rows) {
		double dist = sqrt(pow(abs((double) x - (double) mat.cols / 2), 2) + 
					   pow(abs((double) y - (double) mat.rows / 2), 2));
		dist /= max_dist;
		int alpha = 255;
		if (dist > 0.7) {
			alpha = (int) 255 - (dist-0.7)*1270;
			if (alpha < 0) {
				alpha = 0;
			}
		}
		mat.ptr(y)[4*x + 3] = alpha;
	}
} 

__global__ void subt(cv::gpu::PtrStepSz<uchar> mat, cv::gpu::PtrStepSz<uchar> out) 
{ 
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < mat.cols && y < mat.rows) {
		out.ptr(y)[4*x] = 255 - mat.ptr(y)[4*x];
		out.ptr(y)[4*x + 1] = 255 - mat.ptr(y)[4*x + 1];
		out.ptr(y)[4*x + 2] = 255 - mat.ptr(y)[4*x + 2];
		out.ptr(y)[4*x + 3] = 255 - mat.ptr(y)[4*x + 3];
	}
} 

extern "C"
void alphaKernelCaller(cv::gpu::PtrStepSz<uchar> & x) 
{
	dim3 dimBlock(16, 16);
	dim3 dimGrid((x.cols + (dimBlock.x - 1)) / dimBlock.x, (x.rows + (dimBlock.y - 1)) / dimBlock.y);
	double maxDist = sqrt(pow((double) x.cols/2.0, 2) + pow((double) x.rows/2.0, 2));
   alphaKernel<<<dimGrid,dimBlock>>>(x, maxDist);
   cudaDeviceSynchronize();
} 

extern "C"
void diff(cv::gpu::PtrStepSz<uchar> & x, cv::gpu::PtrStepSz<uchar> & out) {
	dim3 dimBlock(16, 16);
	dim3 dimGrid((x.cols + (dimBlock.x - 1)) / dimBlock.x, (x.rows + (dimBlock.y - 1)) / dimBlock.y);
    subt<<<dimGrid,dimBlock>>>(x, out);
   cudaDeviceSynchronize();
}

#endif
