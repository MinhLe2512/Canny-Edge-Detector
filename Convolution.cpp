#include "Convolution.h"

std::vector<float> Convolution::getKernel() {
	return kernel;
}
void Convolution::setKernel(std::vector<float> kernel, unsigned int kWidth, unsigned int kHeight) {
	this->kernelWidth = kWidth;
	this->kernelHeight = kHeight;
	this->kernel = kernel;
}

void Convolution::doConvolution(const cv::Mat& srcImg, cv::Mat& dstImg) {
	int rows = srcImg.rows, cols = srcImg.cols;

	cv::Mat cloneImg = srcImg.clone();
	//Diretion x, y
	//Ex: kernel 3x3 -> dx = { -1, -1, -1, 0, 0, 0, 1, ,1, 1 }, dy = { -1, 0, 1, -1, 0, 1, -1, 0, 1}
	std::vector<int> dx, dy;
	for (int i = 0; i < kernelHeight; i++) {
		for (int j = 0; j < kernelWidth; j++) {
			dx.push_back(i - kernelHeight / 2);
			dy.push_back(j - kernelWidth / 2);
		}
	}
	dstImg = cv::Mat(rows - kernelHeight + 1, cols - kernelWidth + 1, CV_32FC1, cv::Scalar(0.0));

	for (int i = 0; i < dstImg.rows; i++) {
		float* pData = dstImg.ptr<float>(i);
		for (int j = 0; j < dstImg.cols; j++) {
			int iSrc = i + kernelWidth / 2, jSrc = j + kernelHeight / 2;
			float sum = 0.0;
			for (int data = 0; data < kernelHeight * kernelWidth; data++) {
				float dataImg = cloneImg.ptr<uchar>(iSrc - dx[data])[jSrc - dy[data]] * 1.0;
				float dataKernel = kernel[(dx[data] + kernelHeight / 2) * kernelHeight + dy[data] + kernelWidth / 2] * 1.0;
				sum += dataImg * dataKernel;
			}
			pData[j] = sum;
		}
	}
}
