#pragma once
#ifndef  _CONVOLUTION_H_
#define _CONVOLUTION_H_
#include <opencv2\core\mat.hpp>
#include <iostream>
#include <vector>
class Convolution {
private:
	std::vector<float> kernel;
	unsigned int kernelWidth;
	unsigned int kernelHeight;
public:
	//Get kernel
	std::vector<float> getKernel();
	//Set kernel
	void setKernel(std::vector<float> kernel, unsigned int kWidth, unsigned int kHeight);
	//Do convolution with kernel
	void doConvolution(const cv::Mat& srcImg, cv::Mat& dstImg);
	//Constructor & Destructor
	Convolution() {
		kernel = std::vector<float>(0, 0);
		kernelWidth = kernelHeight = 0;
	};
	~Convolution() {};
};

#endif // ! _CONVOLUTION_H_